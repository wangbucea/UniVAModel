import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
from typing import List, Tuple, Dict
import seaborn as sns
from VLAModel_224 import VLAModel_224
from vla_dataset_loader import create_vla_dataloader
import torchvision.transforms as transforms

class FeatureHeatmapValidator:
    """
    DINOv2特征热力图和DiT模型视觉特征热力图验证器
    """
    
    def __init__(self, model_path: str, dataset_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.dataset_path = dataset_path
        
        # 初始化模型
        self.model = self._load_model()
        
        # 初始化数据加载器
        self.dataloader = self._init_dataloader()
        
        # 创建输出目录
        self.output_dir = "feature_heatmap_validation"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 注册钩子函数用于提取中间特征
        self.features = {}
        self._register_hooks()
    
    def _load_model(self):
        """加载预训练的VLA模型"""
        model = VLAModel_224(
            num_det_classes=10,
            num_seg_classes=2,
            action_dim=7,
            seq_len=5,
            use_dual_view=True
        )
        
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"模型权重已从 {self.model_path} 加载")
        else:
            print(f"警告: 模型权重文件 {self.model_path} 不存在，使用随机初始化权重")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _init_dataloader(self):
        """初始化数据加载器"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataloader = create_vla_dataloader(
            dataset_path=self.dataset_path,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            transform=transform,
            sequence_length=5,
            stride=1
        )
        return dataloader
    
    def _register_hooks(self):
        """注册钩子函数以提取中间特征"""
        def get_activation(name):
            def hook(model, input, output):
                # 处理Hugging Face transformers的输出格式
                if hasattr(output, 'last_hidden_state'):
                    # BaseModelOutputWithPooling格式
                    self.features[name] = output.last_hidden_state.detach()
                elif hasattr(output, 'hidden_states'):
                    # 如果有hidden_states属性
                    self.features[name] = output.hidden_states[-1].detach()
                elif isinstance(output, tuple):
                    # 元组格式，取第一个元素
                    if hasattr(output[0], 'detach'):
                        self.features[name] = output[0].detach()
                    else:
                        # 如果第一个元素也不是tensor，尝试提取last_hidden_state
                        if hasattr(output[0], 'last_hidden_state'):
                            self.features[name] = output[0].last_hidden_state.detach()
                        else:
                            print(f"Warning: Cannot extract features from {name}, output type: {type(output[0])}")
                elif hasattr(output, 'detach'):
                    # 标准tensor格式
                    self.features[name] = output.detach()
                else:
                    print(f"Warning: Cannot extract features from {name}, output type: {type(output)}")
            return hook
        
        # 注册DINOv2特征提取钩子
        if hasattr(self.model.visual_encoder, 'dinov2_view1'):
            # 双视角模式
            self.model.visual_encoder.dinov2_view1.register_forward_hook(
                get_activation('dinov2_view1_features')
            )
            self.model.visual_encoder.dinov2_view2.register_forward_hook(
                get_activation('dinov2_view2_features')
            )
            self.model.visual_encoder.feature_adapter_view1.register_forward_hook(
                get_activation('adapted_view1_features')
            )
            self.model.visual_encoder.feature_adapter_view2.register_forward_hook(
                get_activation('adapted_view2_features')
            )
            self.model.visual_encoder.view_fusion.register_forward_hook(
                get_activation('fused_features')
            )
        else:
            # 单视角模式
            self.model.visual_encoder.dinov2.register_forward_hook(
                get_activation('dinov2_features')
            )
            self.model.visual_encoder.feature_adapter.register_forward_hook(
                get_activation('adapted_features')
            )
        
        # 注册DiT模型相关特征钩子
        if hasattr(self.model, 'dit_action_head'):
            self.model.dit_action_head.register_forward_hook(
                get_activation('dit_input_features')
            )
        
        # 注册统一Transformer特征钩子
        if hasattr(self.model, 'unified_transformer'):
            self.model.unified_transformer.register_forward_hook(
                get_activation('unified_transformer_features')
            )
    
    def _denormalize_image(self, tensor_image):
        """反归一化图像用于可视化"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        if tensor_image.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        
        denorm_image = tensor_image * std + mean
        denorm_image = torch.clamp(denorm_image, 0, 1)
        return denorm_image
    
    def _create_heatmap(self, features, spatial_shape=None, title="Feature Heatmap"):
        """
        创建特征热力图
        Args:
            features: [B, N, C] 或 [B, H, W, C] 特征张量
            spatial_shape: (H, W) 空间形状，如果features是展平的
            title: 热力图标题
        Returns:
            heatmap: numpy数组格式的热力图
        """
        if len(features.shape) == 3:  # [B, N, C]
            B, N, C = features.shape
            if spatial_shape is not None:
                H, W = spatial_shape
                if N == H * W:
                    # 重塑为空间形状
                    features = features.view(B, H, W, C)
                else:
                    # 如果是多尺度特征，取平均
                    features = features.mean(dim=1, keepdim=True)  # [B, 1, C]
                    features = features.view(B, 1, 1, C)
                    H, W = 1, 1
            else:
                # 假设是正方形
                H = W = int(np.sqrt(N))
                if H * W != N:
                    # 如果不是完全平方数，取前H*W个特征
                    H = W = int(np.sqrt(N))
                    features = features[:, :H*W, :].view(B, H, W, C)
                else:
                    features = features.view(B, H, W, C)
        
        # 取第一个batch和通道维度的平均值
        if len(features.shape) == 4:  # [B, H, W, C]
            heatmap = features[0].mean(dim=-1).cpu().numpy()  # [H, W]
        else:
            heatmap = features[0].cpu().numpy()
        
        # 归一化到0-1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _overlay_heatmap_on_image(self, image, heatmap, alpha=0.6):
        """
        将热力图叠加到原始图像上
        Args:
            image: 原始图像 [H, W, 3]
            heatmap: 特征热力图 [H, W]
            alpha: 热力图透明度
        Returns:
            overlayed_image: 叠加后的图像
        """
        # 确保图像和热力图尺寸匹配
        if image.shape[:2] != heatmap.shape:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # 将热力图转换为彩色图像
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 归一化图像到0-255范围
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # 叠加热力图
        overlayed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    def _visualize_dinov2_features_with_overlay(self, sample_idx, images, save_path):
        """可视化DINOv2提取的特征并叠加到原始图像上"""
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f'DINOv2 Feature Heatmaps with Overlay - Sample {sample_idx}', fontsize=16)
        
        # 原始图像
        original_images = []
        for view_idx in range(2):
            original_img = self._denormalize_image(images[view_idx][0, 0])  # 取第一帧
            original_img = original_img.permute(1, 2, 0).cpu().numpy()
            original_images.append(original_img)
            
            axes[0, view_idx].imshow(original_img)
            axes[0, view_idx].set_title(f'Original Image View {view_idx + 1}')
            axes[0, view_idx].axis('off')
        
        # DINOv2原始特征热力图
        if 'dinov2_view1_features' in self.features:
            for view_idx, feature_key in enumerate(['dinov2_view1_features', 'dinov2_view2_features']):
                if feature_key in self.features:
                    features = self.features[feature_key]
                    # DINOv2输出格式: [B, N+1, C]，移除CLS token
                    patch_features = features[:, 1:, :]  # [B, N, C]
                    heatmap = self._create_heatmap(patch_features, spatial_shape=(16, 16))
                    
                    # 纯热力图
                    im1 = axes[1, view_idx].imshow(heatmap, cmap='viridis', interpolation='bilinear')
                    axes[1, view_idx].set_title(f'DINOv2 Raw Features View {view_idx + 1}')
                    axes[1, view_idx].axis('off')
                    plt.colorbar(im1, ax=axes[1, view_idx], fraction=0.046, pad=0.04)
                    
                    # 叠加到原始图像
                    overlayed = self._overlay_heatmap_on_image(original_images[view_idx], heatmap, alpha=0.5)
                    axes[2, view_idx].imshow(overlayed)
                    axes[2, view_idx].set_title(f'DINOv2 Features Overlay View {view_idx + 1}')
                    axes[2, view_idx].axis('off')
        
        # 适配后的特征热力图
        if 'adapted_view1_features' in self.features:
            for view_idx, feature_key in enumerate(['adapted_view1_features', 'adapted_view2_features']):
                if feature_key in self.features:
                    features = self.features[feature_key]
                    heatmap = self._create_heatmap(features, spatial_shape=(16, 16))
                    
                    # 纯热力图
                    im2 = axes[1, view_idx + 2].imshow(heatmap, cmap='plasma', interpolation='bilinear')
                    axes[1, view_idx + 2].set_title(f'Adapted Features View {view_idx + 1}')
                    axes[1, view_idx + 2].axis('off')
                    plt.colorbar(im2, ax=axes[1, view_idx + 2], fraction=0.046, pad=0.04)
                    
                    # 叠加到原始图像
                    overlayed = self._overlay_heatmap_on_image(original_images[view_idx], heatmap, alpha=0.5)
                    axes[2, view_idx + 2].imshow(overlayed)
                    axes[2, view_idx + 2].set_title(f'Adapted Features Overlay View {view_idx + 1}')
                    axes[2, view_idx + 2].axis('off')
        
        # 融合特征
        if 'fused_features' in self.features:
            features = self.features['fused_features']
            heatmap = self._create_heatmap(features, spatial_shape=(16, 16))
            
            # 纯热力图
            im3 = axes[3, 0].imshow(heatmap, cmap='inferno', interpolation='bilinear')
            axes[3, 0].set_title('Fused Features')
            axes[3, 0].axis('off')
            plt.colorbar(im3, ax=axes[3, 0], fraction=0.046, pad=0.04)
            
            # 叠加到两个视角的图像
            for view_idx in range(2):
                overlayed = self._overlay_heatmap_on_image(original_images[view_idx], heatmap, alpha=0.5)
                axes[3, view_idx + 1].imshow(overlayed)
                axes[3, view_idx + 1].set_title(f'Fused Features Overlay View {view_idx + 1}')
                axes[3, view_idx + 1].axis('off')
        
        # 隐藏未使用的子图
        for i in range(4):
            for j in range(4):
                if (i == 0 and j >= 2) or (i == 3 and j == 3):
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_multi_scale_overlay(self, sample_idx, images, save_path):
        """创建多尺度特征热力图叠加可视化"""
        if not any(key in self.features for key in ['dinov2_view1_features', 'adapted_view1_features', 'fused_features']):
            print("没有找到可用的特征进行可视化")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Multi-Scale Feature Overlay - Sample {sample_idx}', fontsize=16)
        
        # 准备原始图像
        original_images = []
        for view_idx in range(2):
            original_img = self._denormalize_image(images[view_idx][0, 0])
            original_img = original_img.permute(1, 2, 0).cpu().numpy()
            original_images.append(original_img)
        
        # 不同透明度的叠加效果
        alphas = [0.3, 0.5, 0.7]
        
        for alpha_idx, alpha in enumerate(alphas):
            # 使用融合特征作为示例
            if 'fused_features' in self.features:
                features = self.features['fused_features']
                heatmap = self._create_heatmap(features, spatial_shape=(16, 16))
                
                # 第一个视角
                overlayed = self._overlay_heatmap_on_image(original_images[0], heatmap, alpha=alpha)
                axes[0, alpha_idx].imshow(overlayed)
                axes[0, alpha_idx].set_title(f'View 1 - Alpha {alpha}')
                axes[0, alpha_idx].axis('off')
                
                # 第二个视角
                overlayed = self._overlay_heatmap_on_image(original_images[1], heatmap, alpha=alpha)
                axes[1, alpha_idx].imshow(overlayed)
                axes[1, alpha_idx].set_title(f'View 2 - Alpha {alpha}')
                axes[1, alpha_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def validate_sample(self, sample_idx=0):
        """验证单个样本的特征热力图"""
        print(f"正在验证样本 {sample_idx}...")
        
        # 获取数据样本
        for i, batch in enumerate(self.dataloader):
            if i == sample_idx:
                break
        else:
            print(f"样本索引 {sample_idx} 超出数据集范围")
            return
        
        # 准备输入数据
        images = [batch['images_view1'].to(self.device), batch['images_view2'].to(self.device)]
        input_actions = batch.get('current_state', torch.zeros(1, 7)).to(self.device)
        
        # 清空之前的特征
        self.features.clear()
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(images, input_actions=input_actions)
        
        # 生成可视化
        sample_dir = os.path.join(self.output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. 原始DINOv2特征可视化
        dinov2_path = os.path.join(sample_dir, "dinov2_features.png")
        self._visualize_dinov2_features(sample_idx, images, dinov2_path)
        print(f"DINOv2特征热力图已保存到: {dinov2_path}")
        
        # 2. 带叠加效果的DINOv2特征可视化
        overlay_path = os.path.join(sample_dir, "dinov2_features_overlay.png")
        self._visualize_dinov2_features_with_overlay(sample_idx, images, overlay_path)
        print(f"DINOv2特征叠加图已保存到: {overlay_path}")
        
        # 3. 多尺度叠加效果
        multi_scale_path = os.path.join(sample_dir, "multi_scale_overlay.png")
        self._create_multi_scale_overlay(sample_idx, images, multi_scale_path)
        print(f"多尺度叠加图已保存到: {multi_scale_path}")
        
        # 4. DiT输入特征可视化
        dit_path = os.path.join(sample_dir, "dit_input_features.png")
        self._visualize_dit_input_features(sample_idx, dit_path)
        print(f"DiT输入特征热力图已保存到: {dit_path}")
        
        # 5. 特征对比
        comparison_path = os.path.join(sample_dir, "feature_comparison.png")
        self._create_feature_comparison(sample_idx, comparison_path)
        print(f"特征对比图已保存到: {comparison_path}")
        
        # 6. 保存特征数据
        features_data = {}
        for name, features in self.features.items():
            if isinstance(features, torch.Tensor):
                features_data[name] = {
                    'shape': list(features.shape),
                    'mean': features.mean().item(),
                    'std': features.std().item(),
                    'min': features.min().item(),
                    'max': features.max().item()
                }
        
        import json
        stats_path = os.path.join(sample_dir, "feature_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(features_data, f, indent=2)
        print(f"特征统计信息已保存到: {stats_path}")
        
        print(f"样本 {sample_idx} 验证完成！")
        return features_data
    
    def validate_multiple_samples(self, num_samples=5):
        """验证多个样本"""
        print(f"开始验证 {num_samples} 个样本...")
        
        all_stats = []
        for i in range(min(num_samples, len(self.dataloader))):
            stats = self.validate_sample(i)
            if stats:
                all_stats.append(stats)
        
        # 创建汇总报告
        self._create_summary_report(all_stats)
        print(f"验证完成！结果保存在 {self.output_dir} 目录中")
    
    def _create_summary_report(self, all_stats):
        """创建汇总报告"""
        if not all_stats:
            return
        
        # 计算所有样本的统计信息
        summary = {}
        feature_names = all_stats[0].keys()
        
        for feature_name in feature_names:
            means = [stats[feature_name]['mean'] for stats in all_stats]
            stds = [stats[feature_name]['std'] for stats in all_stats]
            
            summary[feature_name] = {
                'mean_avg': np.mean(means),
                'mean_std': np.std(means),
                'std_avg': np.mean(stds),
                'std_std': np.std(stds),
                'shape': all_stats[0][feature_name]['shape']
            }
        
        # 保存汇总统计
        import json
        summary_path = os.path.join(self.output_dir, "summary_stats.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 创建汇总可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Statistics Summary Across Samples', fontsize=16)
        
        names = list(summary.keys())
        mean_avgs = [summary[name]['mean_avg'] for name in names]
        std_avgs = [summary[name]['std_avg'] for name in names]
        
        # 平均激活值
        axes[0, 0].bar(range(len(names)), mean_avgs)
        axes[0, 0].set_title('Average Feature Activations')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 平均标准差
        axes[0, 1].bar(range(len(names)), std_avgs)
        axes[0, 1].set_title('Average Feature Standard Deviations')
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 特征稳定性（跨样本的标准差）
        mean_stds = [summary[name]['mean_std'] for name in names]
        axes[1, 0].bar(range(len(names)), mean_stds)
        axes[1, 0].set_title('Feature Stability (Cross-sample Std)')
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 特征维度信息
        dims_text = []
        for name in names:
            shape = summary[name]['shape']
            dims_text.append(f"{name}: {shape}")
        
        axes[1, 1].text(0.1, 0.9, '\n'.join(dims_text), 
                       transform=axes[1, 1].transAxes, 
                       fontsize=9, verticalalignment='top')
        axes[1, 1].set_title('Feature Dimensions')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        summary_plot_path = os.path.join(self.output_dir, "summary_visualization.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"汇总报告已保存到: {summary_path}")
        print(f"汇总可视化已保存到: {summary_plot_path}")


def main():
    """主函数"""
    # 配置参数
    model_path = "path/to/your/model.pth"  # 替换为您的模型路径
    dataset_path = "path/to/your/dataset"   # 替换为您的数据集路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建验证器
    validator = FeatureHeatmapValidator(
        model_path=model_path,
        dataset_path=dataset_path,
        device=device
    )
    
    # 验证单个样本
    print("=== 验证单个样本 ===")
    validator.validate_sample(sample_idx=0)
    
    # 验证多个样本
    print("\n=== 验证多个样本 ===")
    validator.validate_multiple_samples(num_samples=5)
    
    print("\n特征热力图验证完成！")


if __name__ == "__main__":
    main()
