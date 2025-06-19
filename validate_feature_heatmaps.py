import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from typing import List, Tuple, Optional, Dict, Any
import math
from VLAModel import DualViewDINOv2VisualEncoder, DINOv2VisualEncoder
from vla_dataset_loader import VLADatasetLoader

class VLADINOv2Visualizer:
    """
    VLA模型中DINOv2模块的特征可视化工具
    支持单视角和双视角DINOv2编码器的特征热力图生成
    """
    
    def __init__(self, 
                 encoder_type='single',  # 'single' or 'dual'
                 model_name="facebook/dinov2-base",
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 image_size=(480, 640)):
        """
        初始化VLA DINOv2可视化器
        
        Args:
            encoder_type: 编码器类型，'single'表示DINOv2VisualEncoder，'dual'表示DualViewDINOv2VisualEncoder
            model_name: DINOv2模型名称
            device: 计算设备
            image_size: 图像尺寸 (height, width)
        """
        self.device = device
        self.encoder_type = encoder_type
        self.image_size = image_size
        
        # 初始化对应的编码器
        if encoder_type == 'single':
            self.visual_encoder = DINOv2VisualEncoder(
                model_name=model_name,
                freeze_backbone=False,
                image_size=image_size
            ).to(device)
        elif encoder_type == 'dual':
            self.visual_encoder = DualViewDINOv2VisualEncoder(
                model_name=model_name,
                freeze_backbone=False,
                image_size=image_size
            ).to(device)
        else:
            raise ValueError("encoder_type must be 'single' or 'dual'")
        
        self.visual_encoder.eval()
        
        # 存储中间特征的钩子
        self.features = {}
        self.hooks = []
        
        # 图像预处理参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def register_hooks(self):
        """注册钩子函数以提取中间特征"""
        def get_activation(name):
            def hook(model, input, output):
                if hasattr(output, 'last_hidden_state'):
                    # DINOv2模型输出
                    self.features[name] = output.last_hidden_state.detach()
                else:
                    self.features[name] = output.detach()
            return hook
        
        # 清除之前的钩子
        self.clear_hooks()
        
        if self.encoder_type == 'single':
            # 单视角编码器的钩子
            self.hooks.append(
                self.visual_encoder.dinov2.register_forward_hook(
                    get_activation('dinov2_output')
                )
            )
            self.hooks.append(
                self.visual_encoder.feature_adapter.register_forward_hook(
                    get_activation('adapted_features')
                )
            )
        elif self.encoder_type == 'dual':
            # 双视角编码器的钩子
            self.hooks.append(
                self.visual_encoder.dinov2_view1.register_forward_hook(
                    get_activation('dinov2_view1_output')
                )
            )
            self.hooks.append(
                self.visual_encoder.dinov2_view2.register_forward_hook(
                    get_activation('dinov2_view2_output')
                )
            )
            self.hooks.append(
                self.visual_encoder.feature_adapter_view1.register_forward_hook(
                    get_activation('adapted_features_view1')
                )
            )
            self.hooks.append(
                self.visual_encoder.feature_adapter_view2.register_forward_hook(
                    get_activation('adapted_features_view2')
                )
            )
            self.hooks.append(
                self.visual_encoder.view_fusion.register_forward_hook(
                    get_activation('fused_features')
                )
            )
    
    def clear_hooks(self):
        """清除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
    
    def denormalize_image(self, tensor_image):
        """反归一化图像用于显示"""
        denorm_image = tensor_image * self.std + self.mean
        denorm_image = torch.clamp(denorm_image, 0, 1)
        return denorm_image
    
    def extract_patch_features(self, images):
        """
        提取patch级别的特征
        
        Args:
            images: 输入图像，格式根据encoder_type而定
                   - single: [B, T, C, H, W]
                   - dual: [images_view1, images_view2] 每个都是 [B, T, C, H, W]
        
        Returns:
            features_dict: 包含各层特征的字典
        """
        self.register_hooks()
        
        with torch.no_grad():
            if self.encoder_type == 'single':
                # 单视角处理
                Vfeatures_list, spatial_shapes, level_start_index, valid_ratios = self.visual_encoder(images)
            else:
                # 双视角处理
                Vfeatures_list, spatial_shapes, level_start_index, valid_ratios = self.visual_encoder(images)
        
        return {
            'intermediate_features': self.features.copy(),
            'output_features': Vfeatures_list,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index
        }
    
    def create_attention_heatmap(self, features, patch_h, patch_w, method='mean'):
        """
        从patch特征创建注意力热力图
        
        Args:
            features: [B, N, D] patch特征
            patch_h, patch_w: patch的空间维度
            method: 聚合方法 ('mean', 'max', 'norm')
        
        Returns:
            heatmap: [B, patch_h, patch_w] 热力图
        """
        B, N, D = features.shape
        
        if method == 'mean':
            # 对特征维度求平均
            attention_scores = features.mean(dim=-1)  # [B, N]
        elif method == 'max':
            # 取特征维度的最大值
            attention_scores = features.max(dim=-1)[0]  # [B, N]
        elif method == 'norm':
            # 使用L2范数
            attention_scores = torch.norm(features, dim=-1)  # [B, N]
        else:
            raise ValueError("method must be 'mean', 'max', or 'norm'")
        
        # 重塑为空间维度
        if N == patch_h * patch_w:
            heatmap = attention_scores.view(B, patch_h, patch_w)
        else:
            # 如果patch数量不匹配，进行截断或填充
            target_N = patch_h * patch_w
            if N > target_N:
                attention_scores = attention_scores[:, :target_N]
            else:
                padding = torch.zeros(B, target_N - N, device=features.device)
                attention_scores = torch.cat([attention_scores, padding], dim=1)
            heatmap = attention_scores.view(B, patch_h, patch_w)
        
        return heatmap
    
    def overlay_heatmap_on_image(self, image, heatmap, alpha=0.6, colormap='jet'):
        """
        将热力图叠加到原始图像上
        
        Args:
            image: [H, W, 3] 原始图像 (numpy array, 0-1范围)
            heatmap: [H, W] 热力图 (numpy array)
            alpha: 热力图透明度
            colormap: 颜色映射
        
        Returns:
            overlay_image: [H, W, 3] 叠加后的图像
        """
        # 归一化热力图
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 应用颜色映射
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_norm)[:, :, :3]  # 移除alpha通道
        
        # 叠加图像
        overlay_image = (1 - alpha) * image + alpha * heatmap_colored
        
        return np.clip(overlay_image, 0, 1)
    
    def visualize_features_from_dataset(self, 
                                      dataset_path,
                                      sample_idx=0,
                                      time_step=0,
                                      save_dir='./vla_dinov2_visualizations',
                                      methods=['mean', 'norm']):
        """
        从数据集中可视化特征
        
        Args:
            dataset_path: 数据集路径
            sample_idx: 样本索引
            time_step: 时间步索引
            save_dir: 保存目录
            methods: 特征聚合方法列表
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载数据
        dataset_loader = VLADatasetLoader(
            data_dir=dataset_path,
            batch_size=1,
            image_size=self.image_size,
            num_workers=0
        )
        
        dataloader = dataset_loader.get_dataloader()
        
        # 获取样本
        for i, batch in enumerate(dataloader):
            if i == sample_idx:
                break
        else:
            raise ValueError(f"Sample index {sample_idx} not found in dataset")
        
        if self.encoder_type == 'single':
            images = batch['images'].to(self.device)  # [B, T, C, H, W]
            self._visualize_single_view(images, time_step, save_dir, methods)
        else:
            images_view1 = batch['images_view1'].to(self.device)
            images_view2 = batch['images_view2'].to(self.device)
            images = [images_view1, images_view2]
            self._visualize_dual_view(images, time_step, save_dir, methods)
    
    def _visualize_single_view(self, images, time_step, save_dir, methods):
        """可视化单视角特征"""
        B, T, C, H, W = images.shape
        
        # 提取特征
        features_dict = self.extract_patch_features(images)
        
        # 获取原始图像
        original_image = self.denormalize_image(images[0, time_step]).cpu()
        original_image_np = original_image.permute(1, 2, 0).numpy()
        
        # 保存原始图像
        plt.figure(figsize=(8, 6))
        plt.imshow(original_image_np)
        plt.title(f'Original Image (Time Step {time_step})')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'original_image_t{time_step}.png'), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # 可视化DINOv2原始特征
        if 'dinov2_output' in features_dict['intermediate_features']:
            dinov2_features = features_dict['intermediate_features']['dinov2_output']
            # 移除CLS token
            patch_features = dinov2_features[0, 1:, :]  # [N, 768]
            
            # 计算patch维度
            num_patches = patch_features.shape[0]
            patch_h = patch_w = int(math.sqrt(num_patches))
            
            if patch_h * patch_w != num_patches:
                aspect_ratio = W / H
                patch_h = int(math.sqrt(num_patches / aspect_ratio))
                patch_w = int(num_patches / patch_h)
                if patch_h * patch_w != num_patches:
                    patch_h = patch_w = int(math.sqrt(num_patches))
                    patch_features = patch_features[:patch_h*patch_w, :]
            
            for method in methods:
                heatmap = self.create_attention_heatmap(
                    patch_features.unsqueeze(0), patch_h, patch_w, method
                )[0].cpu().numpy()
                
                # 上采样热力图到原始图像尺寸
                heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
                
                # 创建叠加图像
                overlay_image = self.overlay_heatmap_on_image(
                    original_image_np, heatmap_resized
                )
                
                # 保存可视化结果
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(original_image_np)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                im1 = axes[1].imshow(heatmap_resized, cmap='jet')
                axes[1].set_title(f'DINOv2 Features ({method})')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1])
                
                axes[2].imshow(overlay_image)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'dinov2_features_{method}_t{time_step}.png'), 
                           bbox_inches='tight', dpi=150)
                plt.close()
        
        # 可视化适配后的特征
        if 'adapted_features' in features_dict['intermediate_features']:
            adapted_features = features_dict['intermediate_features']['adapted_features']
            
            # 计算patch维度
            num_patches = adapted_features.shape[1]
            patch_h = patch_w = int(math.sqrt(num_patches))
            
            if patch_h * patch_w != num_patches:
                aspect_ratio = W / H
                patch_h = int(math.sqrt(num_patches / aspect_ratio))
                patch_w = int(num_patches / patch_h)
                if patch_h * patch_w != num_patches:
                    patch_h = patch_w = int(math.sqrt(num_patches))
            
            for method in methods:
                heatmap = self.create_attention_heatmap(
                    adapted_features, patch_h, patch_w, method
                )[0].cpu().numpy()
                
                # 上采样热力图到原始图像尺寸
                heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
                
                # 创建叠加图像
                overlay_image = self.overlay_heatmap_on_image(
                    original_image_np, heatmap_resized
                )
                
                # 保存可视化结果
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(original_image_np)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                im1 = axes[1].imshow(heatmap_resized, cmap='jet')
                axes[1].set_title(f'Adapted Features ({method})')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1])
                
                axes[2].imshow(overlay_image)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'adapted_features_{method}_t{time_step}.png'), 
                           bbox_inches='tight', dpi=150)
                plt.close()
    
    def _visualize_dual_view(self, images, time_step, save_dir, methods):
        """可视化双视角特征"""
        images_view1, images_view2 = images
        B, T, C, H, W = images_view1.shape
        
        # 提取特征
        features_dict = self.extract_patch_features(images)
        
        # 获取原始图像
        original_image_view1 = self.denormalize_image(images_view1[0, time_step]).cpu()
        original_image_view2 = self.denormalize_image(images_view2[0, time_step]).cpu()
        original_image_view1_np = original_image_view1.permute(1, 2, 0).numpy()
        original_image_view2_np = original_image_view2.permute(1, 2, 0).numpy()
        
        # 保存原始图像
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].imshow(original_image_view1_np)
        axes[0].set_title(f'View 1 - Original Image (Time Step {time_step})')
        axes[0].axis('off')
        
        axes[1].imshow(original_image_view2_np)
        axes[1].set_title(f'View 2 - Original Image (Time Step {time_step})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'original_images_dual_t{time_step}.png'), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # 可视化各个视角的DINOv2特征
        for view_idx, (view_name, original_image_np) in enumerate([
            ('view1', original_image_view1_np), 
            ('view2', original_image_view2_np)
        ]):
            feature_key = f'dinov2_{view_name}_output'
            if feature_key in features_dict['intermediate_features']:
                dinov2_features = features_dict['intermediate_features'][feature_key]
                # 移除CLS token
                patch_features = dinov2_features[0, 1:, :]  # [N, 768]
                
                # 计算patch维度
                num_patches = patch_features.shape[0]
                patch_h = patch_w = int(math.sqrt(num_patches))
                
                if patch_h * patch_w != num_patches:
                    aspect_ratio = W / H
                    patch_h = int(math.sqrt(num_patches / aspect_ratio))
                    patch_w = int(num_patches / patch_h)
                    if patch_h * patch_w != num_patches:
                        patch_h = patch_w = int(math.sqrt(num_patches))
                        patch_features = patch_features[:patch_h*patch_w, :]
                
                for method in methods:
                    heatmap = self.create_attention_heatmap(
                        patch_features.unsqueeze(0), patch_h, patch_w, method
                    )[0].cpu().numpy()
                    
                    # 上采样热力图到原始图像尺寸
                    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
                    
                    # 创建叠加图像
                    overlay_image = self.overlay_heatmap_on_image(
                        original_image_np, heatmap_resized
                    )
                    
                    # 保存可视化结果
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    axes[0].imshow(original_image_np)
                    axes[0].set_title(f'{view_name.title()} - Original Image')
                    axes[0].axis('off')
                    
                    im1 = axes[1].imshow(heatmap_resized, cmap='jet')
                    axes[1].set_title(f'{view_name.title()} - DINOv2 Features ({method})')
                    axes[1].axis('off')
                    plt.colorbar(im1, ax=axes[1])
                    
                    axes[2].imshow(overlay_image)
                    axes[2].set_title(f'{view_name.title()} - Overlay')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'dinov2_{view_name}_features_{method}_t{time_step}.png'), 
                               bbox_inches='tight', dpi=150)
                    plt.close()
        
        # 可视化融合后的特征（如果有的话）
        if 'fused_features' in features_dict['intermediate_features']:
            # 注意：融合特征的形状可能与原始patch特征不同
            print("Fused features visualization would require additional processing based on the fusion output format.")
    
    def visualize_custom_images(self, 
                              image_paths,
                              save_dir='./vla_dinov2_custom_visualizations',
                              methods=['mean', 'norm']):
        """
        可视化自定义图像的特征
        
        Args:
            image_paths: 图像路径列表
                        - 单视角: [path1, path2] (两个时间步)
                        - 双视角: [[view1_path1, view1_path2], [view2_path1, view2_path2]]
            save_dir: 保存目录
            methods: 特征聚合方法列表
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.encoder_type == 'single':
            # 单视角处理
            assert len(image_paths) == 2, "Single view requires 2 time steps"
            images = self._load_custom_images_single(image_paths)
            self._visualize_single_view(images, 1, save_dir, methods)  # 可视化第二个时间步
        else:
            # 双视角处理
            assert len(image_paths) == 2 and len(image_paths[0]) == 2, "Dual view requires 2 views with 2 time steps each"
            images = self._load_custom_images_dual(image_paths)
            self._visualize_dual_view(images, 1, save_dir, methods)  # 可视化第二个时间步
    
    def _load_custom_images_single(self, image_paths):
        """加载单视角自定义图像"""
        images_list = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            image = image.resize((self.image_size[1], self.image_size[0]))  # (width, height)
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # [C, H, W]
            # 归一化
            image_tensor = (image_tensor - self.mean.squeeze()) / self.std.squeeze()
            images_list.append(image_tensor)
        
        images = torch.stack(images_list, dim=0).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        return images
    
    def _load_custom_images_dual(self, image_paths):
        """加载双视角自定义图像"""
        view1_paths, view2_paths = image_paths
        
        images_view1 = self._load_custom_images_single(view1_paths)
        images_view2 = self._load_custom_images_single(view2_paths)
        
        return [images_view1, images_view2]
    
    def __del__(self):
        """析构函数，清理钩子"""
        self.clear_hooks()


# 使用示例
if __name__ == "__main__":
    # 单视角DINOv2可视化
    print("创建单视角DINOv2可视化器...")
    single_visualizer = VLADINOv2Visualizer(
        encoder_type='single',
        model_name="facebook/dinov2-base",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 从数据集可视化
    try:
        single_visualizer.visualize_features_from_dataset(
            dataset_path='./james',  # 替换为您的数据集路径
            sample_idx=0,
            time_step=1,
            save_dir='./single_view_visualizations',
            methods=['mean', 'norm']
        )
        print("单视角可视化完成！")
    except Exception as e:
        print(f"单视角数据集可视化失败: {e}")
    
    # 双视角DINOv2可视化
    print("\n创建双视角DINOv2可视化器...")
    dual_visualizer = VLADINOv2Visualizer(
        encoder_type='dual',
        model_name="facebook/dinov2-base",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 从数据集可视化
    try:
        dual_visualizer.visualize_features_from_dataset(
            dataset_path='./james',  # 替换为您的数据集路径
            sample_idx=0,
            time_step=1,
            save_dir='./dual_view_visualizations',
            methods=['mean', 'norm']
        )
        print("双视角可视化完成！")
    except Exception as e:
        print(f"双视角数据集可视化失败: {e}")
    
    # 自定义图像可视化示例
    # single_visualizer.visualize_custom_images(
    #     image_paths=['path/to/image1.jpg', 'path/to/image2.jpg'],
    #     save_dir='./custom_single_visualizations'
    # )
    
    # dual_visualizer.visualize_custom_images(
    #     image_paths=[
    #         ['path/to/view1_image1.jpg', 'path/to/view1_image2.jpg'],
    #         ['path/to/view2_image1.jpg', 'path/to/view2_image2.jpg']
    #     ],
    #     save_dir='./custom_dual_visualizations'
    # )
