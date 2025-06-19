import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from VLAModel import DualViewDINOv2VisualEncoder, DINOv2VisualEncoder
from vla_dataset_loader import create_vla_dataloader
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class DINOv2FeatureVisualizer:
    """
    DINOv2物体特征可视化器
    专门用于提取和可视化DINOv2模型识别的物体特征
    """
    
    def __init__(self, 
                 model_name: str = "facebook/dinov2-base",
                 dataset_path: str = None,
                 device: str = 'cuda',
                 use_dual_view: bool = True,
                 image_size: Tuple[int, int] = (224, 224)):
        """
        初始化DINOv2特征可视化器
        
        Args:
            model_name: DINOv2模型名称
            dataset_path: 数据集路径
            device: 计算设备
            use_dual_view: 是否使用双视角模式
            image_size: 图像尺寸
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.use_dual_view = use_dual_view
        self.image_size = image_size
        
        # 初始化DINOv2编码器
        self.visual_encoder = self._init_visual_encoder()
        
        # 初始化数据加载器（如果提供了数据集路径）
        self.dataloader = None
        if dataset_path:
            self.dataloader = self._init_dataloader()
        
        # 创建输出目录
        self.output_dir = "dinov2_feature_visualization"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储提取的特征
        self.features = {}
        self._register_hooks()
    
    def _init_visual_encoder(self):
        """初始化视觉编码器"""
        if self.use_dual_view:
            encoder = DualViewDINOv2VisualEncoder(
                model_name=self.model_name,
                freeze_backbone=False,
                image_size=self.image_size
            )
        else:
            encoder = DINOv2VisualEncoder(
                model_name=self.model_name,
                freeze_backbone=False,
                image_size=self.image_size
            )
        
        encoder.to(self.device)
        encoder.eval()
        return encoder
    
    def _init_dataloader(self):
        """初始化数据加载器"""
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
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
                if hasattr(output, 'last_hidden_state'):
                    self.features[name] = output.last_hidden_state.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    if hasattr(output[0], 'detach'):
                        self.features[name] = output[0].detach()
                elif hasattr(output, 'detach'):
                    self.features[name] = output.detach()
            return hook
        
        # 注册不同模式的钩子
        if self.use_dual_view:
            self.visual_encoder.dinov2_view1.register_forward_hook(
                get_activation('dinov2_view1_raw')
            )
            self.visual_encoder.dinov2_view2.register_forward_hook(
                get_activation('dinov2_view2_raw')
            )
            self.visual_encoder.feature_adapter_view1.register_forward_hook(
                get_activation('adapted_view1')
            )
            self.visual_encoder.feature_adapter_view2.register_forward_hook(
                get_activation('adapted_view2')
            )
        else:
            self.visual_encoder.dinov2.register_forward_hook(
                get_activation('dinov2_raw')
            )
            self.visual_encoder.feature_adapter.register_forward_hook(
                get_activation('adapted_features')
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
    
    def _extract_object_features(self, features, method='attention', top_k=5):
        """
        从DINOv2特征中提取物体相关的特征
        
        Args:
            features: [B, N, C] DINOv2特征
            method: 提取方法 ('attention', 'clustering', 'pca')
            top_k: 提取的top-k特征数量
        
        Returns:
            object_features: 物体相关特征
            attention_weights: 注意力权重
        """
        B, N, C = features.shape
        features_np = features[0].cpu().numpy()  # 取第一个batch
        
        if method == 'attention':
            # 使用自注意力机制识别重要区域
            attention_scores = torch.matmul(features[0], features[0].transpose(-2, -1))  # [N, N]
            attention_weights = torch.softmax(attention_scores.mean(dim=1), dim=0)  # [N]
            
            # 选择top-k最重要的patch
            top_indices = torch.topk(attention_weights, top_k).indices
            object_features = features[0][top_indices]  # [top_k, C]
            
        elif method == 'clustering':
            # 使用K-means聚类识别物体区域
            kmeans = KMeans(n_clusters=top_k, random_state=42)
            cluster_labels = kmeans.fit_predict(features_np)
            
            # 计算每个cluster的中心特征
            object_features = []
            attention_weights = torch.zeros(N)
            
            for i in range(top_k):
                cluster_mask = cluster_labels == i
                if cluster_mask.sum() > 0:
                    cluster_features = features_np[cluster_mask]
                    cluster_center = cluster_features.mean(axis=0)
                    object_features.append(cluster_center)
                    attention_weights[cluster_mask] = 1.0 / cluster_mask.sum()
            
            object_features = torch.tensor(np.array(object_features), device=features.device)
            
        elif method == 'pca':
            # 使用PCA降维并识别主要成分
            pca = PCA(n_components=min(top_k, C))
            pca_features = pca.fit_transform(features_np)
            
            # 计算每个patch在主成分上的投影强度
            projection_strength = np.abs(pca_features).sum(axis=1)
            top_indices = np.argsort(projection_strength)[-top_k:]
            
            object_features = features[0][top_indices]
            attention_weights = torch.zeros(N)
            attention_weights[top_indices] = torch.tensor(projection_strength[top_indices])
            attention_weights = attention_weights / attention_weights.sum()
        
        return object_features, attention_weights
    
    def _create_object_heatmap(self, attention_weights, spatial_shape=(16, 16)):
        """
        根据注意力权重创建物体热力图
        
        Args:
            attention_weights: [N] 注意力权重
            spatial_shape: 空间形状
        
        Returns:
            heatmap: [H, W] 热力图
        """
        H, W = spatial_shape
        N = attention_weights.shape[0]
        
        # 确保权重数量与空间大小匹配
        if N != H * W:
            # 如果不匹配，进行插值调整
            current_h = current_w = int(np.sqrt(N))
            if current_h * current_w != N:
                # 截断到最接近的平方数
                current_h = current_w = int(np.sqrt(N))
                attention_weights = attention_weights[:current_h*current_w]
            
            # 重塑并插值到目标尺寸
            heatmap = attention_weights.view(current_h, current_w).cpu().numpy()
            heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap = attention_weights.view(H, W).cpu().numpy()
        
        # 归一化到0-1范围
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _overlay_heatmap_on_image(self, image, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
        """
        将热力图叠加到原始图像上
        
        Args:
            image: 原始图像 [H, W, 3]
            heatmap: 特征热力图 [H, W]
            alpha: 热力图透明度
            colormap: 颜色映射
        
        Returns:
            overlayed_image: 叠加后的图像
        """
        # 确保图像和热力图尺寸匹配
        if image.shape[:2] != heatmap.shape:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # 将热力图转换为彩色图像
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 归一化图像到0-255范围
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # 叠加热力图
        overlayed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    def visualize_object_features(self, 
                                 images: torch.Tensor,
                                 save_path: str = None,
                                 extraction_methods: List[str] = ['attention', 'clustering'],
                                 top_k: int = 5,
                                 alpha: float = 0.6):
        """
        可视化DINOv2提取的物体特征
        
        Args:
            images: 输入图像 [B, T, C, H, W] 或 List[[B, T, C, H, W]] (双视角)
            save_path: 保存路径
            extraction_methods: 特征提取方法列表
            top_k: 提取的top-k特征数量
            alpha: 热力图透明度
        """
        # 清空之前的特征
        self.features.clear()
        
        # 前向传播提取特征
        with torch.no_grad():
            if self.use_dual_view:
                if not isinstance(images, list) or len(images) != 2:
                    raise ValueError("双视角模式需要提供两个视角的图像")
                outputs = self.visual_encoder(images)
            else:
                if isinstance(images, list):
                    images = images[0]  # 取第一个视角
                outputs = self.visual_encoder(images)
        
        # 准备原始图像用于可视化
        if self.use_dual_view:
            original_images = []
            for view_idx in range(2):
                img = self._denormalize_image(images[view_idx][0, 0])  # 取第一帧
                img = img.permute(1, 2, 0).cpu().numpy()
                original_images.append(img)
        else:
            img = self._denormalize_image(images[0, 0])  # 取第一帧
            original_images = [img.permute(1, 2, 0).cpu().numpy()]
        
        # 创建可视化
        num_methods = len(extraction_methods)
        num_views = 2 if self.use_dual_view else 1
        
        fig, axes = plt.subplots(num_methods + 1, num_views * 2, 
                                figsize=(num_views * 8, (num_methods + 1) * 4))
        
        if num_methods == 1:
            axes = axes.reshape(num_methods + 1, -1)
        
        fig.suptitle('DINOv2 Object Feature Visualization', fontsize=16)
        
        # 显示原始图像
        for view_idx in range(num_views):
            axes[0, view_idx * 2].imshow(original_images[view_idx])
            axes[0, view_idx * 2].set_title(f'Original Image View {view_idx + 1}')
            axes[0, view_idx * 2].axis('off')
            
            # 空白第二列
            axes[0, view_idx * 2 + 1].axis('off')
        
        # 对每种提取方法进行可视化
        for method_idx, method in enumerate(extraction_methods):
            if self.use_dual_view:
                # 双视角模式
                for view_idx in range(2):
                    feature_key = f'dinov2_view{view_idx + 1}_raw'
                    if feature_key in self.features:
                        features = self.features[feature_key]
                        # 移除CLS token
                        patch_features = features[:, 1:, :]
                        
                        # 提取物体特征
                        object_features, attention_weights = self._extract_object_features(
                            patch_features, method=method, top_k=top_k
                        )
                        
                        # 创建热力图
                        heatmap = self._create_object_heatmap(attention_weights)
                        
                        # 显示纯热力图
                        im = axes[method_idx + 1, view_idx * 2].imshow(
                            heatmap, cmap='viridis', interpolation='bilinear'
                        )
                        axes[method_idx + 1, view_idx * 2].set_title(
                            f'{method.capitalize()} Heatmap View {view_idx + 1}'
                        )
                        axes[method_idx + 1, view_idx * 2].axis('off')
                        plt.colorbar(im, ax=axes[method_idx + 1, view_idx * 2], 
                                   fraction=0.046, pad=0.04)
                        
                        # 显示叠加图像
                        overlayed = self._overlay_heatmap_on_image(
                            original_images[view_idx], heatmap, alpha=alpha
                        )
                        axes[method_idx + 1, view_idx * 2 + 1].imshow(overlayed)
                        axes[method_idx + 1, view_idx * 2 + 1].set_title(
                            f'{method.capitalize()} Overlay View {view_idx + 1}'
                        )
                        axes[method_idx + 1, view_idx * 2 + 1].axis('off')
            else:
                # 单视角模式
                feature_key = 'dinov2_raw'
                if feature_key in self.features:
                    features = self.features[feature_key]
                    # 移除CLS token
                    patch_features = features[:, 1:, :]
                    
                    # 提取物体特征
                    object_features, attention_weights = self._extract_object_features(
                        patch_features, method=method, top_k=top_k
                    )
                    
                    # 创建热力图
                    heatmap = self._create_object_heatmap(attention_weights)
                    
                    # 显示纯热力图
                    im = axes[method_idx + 1, 0].imshow(
                        heatmap, cmap='viridis', interpolation='bilinear'
                    )
                    axes[method_idx + 1, 0].set_title(f'{method.capitalize()} Heatmap')
                    axes[method_idx + 1, 0].axis('off')
                    plt.colorbar(im, ax=axes[method_idx + 1, 0], fraction=0.046, pad=0.04)
                    
                    # 显示叠加图像
                    overlayed = self._overlay_heatmap_on_image(
                        original_images[0], heatmap, alpha=alpha
                    )
                    axes[method_idx + 1, 1].imshow(overlayed)
                    axes[method_idx + 1, 1].set_title(f'{method.capitalize()} Overlay')
                    axes[method_idx + 1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_from_dataset(self, sample_idx: int = 0, **kwargs):
        """
        从数据集中可视化指定样本的物体特征
        
        Args:
            sample_idx: 样本索引
            **kwargs: 传递给visualize_object_features的其他参数
        """
        if self.dataloader is None:
            raise ValueError("未提供数据集路径，无法从数据集加载样本")
        
        # 获取指定样本
        for i, batch in enumerate(self.dataloader):
            if i == sample_idx:
                break
        else:
            raise IndexError(f"样本索引 {sample_idx} 超出数据集范围")
        
        # 准备输入图像
        if self.use_dual_view:
            images = [
                batch['chest_images'].to(self.device),
                batch['head_images'].to(self.device)
            ]
        else:
            images = batch['chest_images'].to(self.device)
        
        # 设置保存路径
        save_path = os.path.join(self.output_dir, f"sample_{sample_idx}_object_features.png")
        
        # 可视化
        self.visualize_object_features(images, save_path=save_path, **kwargs)
    
    def visualize_from_image_paths(self, 
                                  image_paths: List[str],
                                  **kwargs):
        """
        从图像路径可视化物体特征
        
        Args:
            image_paths: 图像路径列表 (单视角: [path], 双视角: [path1, path2])
            **kwargs: 传递给visualize_object_features的其他参数
        """
        # 加载和预处理图像
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.use_dual_view:
            if len(image_paths) != 2:
                raise ValueError("双视角模式需要提供两个图像路径")
            
            images = []
            for path in image_paths:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
                images.append(img_tensor.to(self.device))
        else:
            if len(image_paths) != 1:
                raise ValueError("单视角模式只需要一个图像路径")
            
            img = Image.open(image_paths[0]).convert('RGB')
            images = transform(img).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, C, H, W]
        
        # 设置保存路径
        save_path = os.path.join(self.output_dir, "custom_image_object_features.png")
        
        # 可视化
        self.visualize_object_features(images, save_path=save_path, **kwargs)


# 使用示例
if __name__ == "__main__":
    # 初始化可视化器
    visualizer = DINOv2FeatureVisualizer(
        model_name="facebook/dinov2-base",
        dataset_path="c:/DiskD/trae_doc/VLA",
        device='cuda',
        use_dual_view=True,
        image_size=(224, 224)
    )
    
    # 从数据集可视化
    visualizer.visualize_from_dataset(
        sample_idx=0,
        extraction_methods=['attention', 'clustering', 'pca'],
        top_k=5,
        alpha=0.6
    )
    
    # 从自定义图像可视化
    # visualizer.visualize_from_image_paths(
    #     image_paths=["path/to/image1.jpg", "path/to/image2.jpg"],
    #     extraction_methods=['attention'],
    #     top_k=3,
    #     alpha=0.5
    # )
