import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
from DITActionHead import DITActionHead, DDPMScheduler
from transformers import Dinov2Model, Dinov2Config
from DeformableTransformer import *
from compute import *

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TripleViewDINOv2VisualEncoder(nn.Module):
    """三视角DINOv2视觉编码器 - 适配224x224输入"""
    def __init__(self, model_name="facebook/dinov2-base", freeze_backbone=False, image_size=(224, 224)):
        super().__init__()
        
        # 三个视角的DINOv2编码器
        self.dinov2_view1 = Dinov2Model.from_pretrained(model_name)
        self.dinov2_view2 = Dinov2Model.from_pretrained(model_name)
        self.dinov2_view3 = Dinov2Model.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.dinov2_view1.parameters():
                param.requires_grad = False
            for param in self.dinov2_view2.parameters():
                param.requires_grad = False
            for param in self.dinov2_view3.parameters():
                param.requires_grad = False
        
        # 特征维度适配
        self.feature_dim = self.dinov2_view1.config.hidden_size  # 768 for base model
        self.feature_adapter_view1 = nn.Linear(self.feature_dim, 256)
        self.feature_adapter_view2 = nn.Linear(self.feature_dim, 256)
        self.feature_adapter_view3 = nn.Linear(self.feature_dim, 256)
        
        # 多尺度特征投影
        self.level_embed = nn.Parameter(torch.Tensor(4, 256))
        nn.init.normal_(self.level_embed)
        
        # 三视角融合模块 - 使用多头注意力机制
        self.view_fusion_12 = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        self.view_fusion_13 = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        self.view_fusion_23 = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        
        # 全局三视角融合
        self.global_view_fusion = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        
        # 视角标识嵌入
        self.view1_embed = nn.Parameter(torch.randn(1, 1, 256))
        self.view2_embed = nn.Parameter(torch.randn(1, 1, 256))
        self.view3_embed = nn.Parameter(torch.randn(1, 1, 256))
        
        # 融合后的特征投影
        self.fusion_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.layer_norm = nn.LayerNorm(256)
        
        # 三视角权重学习
        self.view_weight_net = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=-1)
        )
        self.layer_norm = nn.LayerNorm(256)
        
    def encode_single_view(self, images, dinov2_model, feature_adapter, view_embed):
        """
        编码单个视角的图像 - 适配224x224输入
        Args:
            images: [B, T, C, H, W] 单个视角的输入图像 (H=W=224)
            dinov2_model: DINOv2模型
            feature_adapter: 特征适配器
            view_embed: 视角嵌入
        Returns:
            Vfeatures_list: List[[B, \sum H_i*W_i, 256]] 每个时间步的多尺度特征
        """
        B, T, C, H, W = images.shape
        assert H == W == 224, f"Expected 224x224 input, got {H}x{W}"
        
        Vfeatures_list = []
        
        # 分别处理每个时间步的图像
        for t in range(T):
            current_images = images[:, t, :, :, :]  # [B, C, H, W]
            
            with torch.no_grad() if hasattr(self, 'freeze_backbone') else torch.enable_grad():
                outputs = dinov2_model(current_images)
                # 获取patch embeddings
                patch_embeddings = outputs.last_hidden_state  # [B, N+1, 768]
                # 移除CLS token
                base_features = patch_embeddings[:, 1:, :]  # [B, N, 768]
            
            # 特征维度适配
            base_features = feature_adapter(base_features)  # [B, N, 256]
            
            # 添加视角嵌入
            base_features = base_features + view_embed.expand(B, base_features.shape[1], -1)
            
            # 对于224x224输入，DINOv2的patch数量是固定的
            # DINOv2-base使用14x14的patch size，所以224x224图像会产生16x16=256个patches
            num_patches = base_features.shape[1]  # 应该是256
            patch_h = patch_w = int(math.sqrt(num_patches))  # 应该是16
            
            # 确保patch数量正确
            if patch_h * patch_w != num_patches:
                patch_h = patch_w = 16  # 强制设置为16x16
                base_features = base_features[:, :patch_h*patch_w, :]
            
            # 重塑为空间特征图
            spatial_features = base_features.view(B, patch_h, patch_w, 256).permute(0, 3, 1, 2)  # [B, 256, 16, 16]
            
            # 生成多尺度特征
            multi_scale_features = []
            
            # 4个尺度的特征
            scales = [1, 2, 4, 8]  # 下采样倍数
            
            for i, scale in enumerate(scales):
                if scale == 1:
                    feat = spatial_features  # [B, 256, 16, 16]
                else:
                    feat = F.avg_pool2d(spatial_features, kernel_size=scale, stride=scale)
                
                # 展平并添加位置编码
                feat_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, 256]
                feat_flat = feat_flat + self.level_embed[i].view(1, 1, -1)
                multi_scale_features.append(feat_flat)
            
            # 拼接所有尺度的特征
            Vfeatures = torch.cat(multi_scale_features, dim=1)  # [B, \sum H_i*W_i, 256]
            Vfeatures_list.append(Vfeatures)
        
        return Vfeatures_list
    
    def forward(self, images):
        """
        Args:
            images[0]: [B, T, C, H, W] 第一个视角的输入图像，H=W=224
            images[1]: [B, T, C, H, W] 第二个视角的输入图像，H=W=224
            images[2]: [B, T, C, H, W] 第三个视角的输入图像，H=W=224
        Returns:
            Vfeatures_list: List[[B, \sum H_i*W_i, 256]] 每个时间步的融合多尺度特征
            spatial_shapes: [n_levels, 2] 每个尺度的空间形状
            level_start_index: [n_levels] 每个尺度的起始索引
            valid_ratios: [B, n_levels, 2] 有效区域比例
        """
        B, T, C, H, W = images[0].shape
        assert H == W == 224, f"Expected 224x224 input, got {H}x{W}"
        assert images[0].shape == images[1].shape == images[2].shape, "All three views must have the same shape"
        
        # 编码三个视角
        Vfeatures_list_view1 = self.encode_single_view(
            images[0], self.dinov2_view1, self.feature_adapter_view1, self.view1_embed
        )
        Vfeatures_list_view2 = self.encode_single_view(
            images[1], self.dinov2_view2, self.feature_adapter_view2, self.view2_embed
        )
        Vfeatures_list_view3 = self.encode_single_view(
            images[2], self.dinov2_view3, self.feature_adapter_view3, self.view3_embed
        )
        
        # 融合三个视角的特征
        fused_Vfeatures_list = []
        for t in range(T):
            view1_features = Vfeatures_list_view1[t]  # [B, N, 256]
            view2_features = Vfeatures_list_view2[t]  # [B, N, 256]
            view3_features = Vfeatures_list_view3[t]  # [B, N, 256]
            
            # 两两交叉注意力融合
            fused_12, _ = self.view_fusion_12(
                query=view1_features,
                key=view2_features,
                value=view2_features
            )
            
            fused_13, _ = self.view_fusion_13(
                query=view1_features,
                key=view3_features,
                value=view3_features
            )
            
            fused_23, _ = self.view_fusion_23(
                query=view2_features,
                key=view3_features,
                value=view3_features
            )
            
            # 将三个视角的特征拼接
            combined_features = torch.cat([fused_12, fused_13, fused_23], dim=-1)  # [B, N, 768]
            
            # 学习三个视角的权重
            view_weights = self.view_weight_net(combined_features)  # [B, N, 3]
            
            # 加权融合三个视角
            weighted_view1 = view1_features * view_weights[:, :, 0:1]
            weighted_view2 = view2_features * view_weights[:, :, 1:2]
            weighted_view3 = view3_features * view_weights[:, :, 2:3]
            
            # 最终融合
            fused_features = weighted_view1 + weighted_view2 + weighted_view3
            enhanced_features = self.layer_norm(fused_features / 3)
            
            # 全局融合
            global_fused, _ = self.global_view_fusion(
                query=enhanced_features,
                key=enhanced_features,
                value=enhanced_features
            )
            
            # 残差连接和投影
            final_features = self.layer_norm(enhanced_features + global_fused)
            final_features = self.fusion_projection(final_features)
            
            fused_Vfeatures_list.append(final_features)
        
        # 计算spatial_shapes和level_start_index（基于224x224输入）
        spatial_shapes = []
        level_start_index = []
        scales = [1, 2, 4, 8]
        
        # 对于224x224输入，基础patch尺寸是16x16
        base_patch_h = base_patch_w = 16
        
        start_idx = 0
        for i, scale in enumerate(scales):
            if scale == 1:
                h, w = base_patch_h, base_patch_w  # 16, 16
            else:
                h, w = max(1, base_patch_h // scale), max(1, base_patch_w // scale)
            
            spatial_shapes.append([h, w])
            level_start_index.append(start_idx)
            start_idx += h * w
        
        # 转换为tensor格式
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=images[0].device)
        level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=images[0].device)
        
        # 有效区域比例（假设全部有效）
        valid_ratios = torch.ones(B, len(scales), 2, device=images[0].device)
        
        return fused_Vfeatures_list, spatial_shapes, level_start_index, valid_ratios

class DualViewDINOv2VisualEncoder(nn.Module):
    """双视角DINOv2视觉编码器 - 适配224x224输入"""
    def __init__(self, model_name="facebook/dinov2-base", freeze_backbone=False, image_size=(224, 224)):
        super().__init__()
        
        # 第一个视角的DINOv2编码器
        self.dinov2_view1 = Dinov2Model.from_pretrained(model_name)
        # 第二个视角的DINOv2编码器
        self.dinov2_view2 = Dinov2Model.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.dinov2_view1.parameters():
                param.requires_grad = False
            for param in self.dinov2_view2.parameters():
                param.requires_grad = False
        
        # 特征维度适配
        self.feature_dim = self.dinov2_view1.config.hidden_size  # 768 for base model
        self.feature_adapter_view1 = nn.Linear(self.feature_dim, 256)
        self.feature_adapter_view2 = nn.Linear(self.feature_dim, 256)
        
        # 多尺度特征投影
        self.level_embed = nn.Parameter(torch.Tensor(4, 256))
        nn.init.normal_(self.level_embed)
        
        # 视角融合模块
        self.view_fusion = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        
        # 视角标识嵌入
        self.view1_embed = nn.Parameter(torch.randn(1, 1, 256))
        self.view2_embed = nn.Parameter(torch.randn(1, 1, 256))
        
        # 融合后的特征投影
        self.fusion_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.layer_norm = nn.LayerNorm(256)
        
    def encode_single_view(self, images, dinov2_model, feature_adapter, view_embed):
        """
        编码单个视角的图像 - 适配224x224输入
        Args:
            images: [B, T, C, H, W] 单个视角的输入图像 (H=W=224)
            dinov2_model: DINOv2模型
            feature_adapter: 特征适配器
            view_embed: 视角嵌入
        Returns:
            Vfeatures_list: List[[B, \sum H_i*W_i, 256]] 每个时间步的多尺度特征
        """
        B, T, C, H, W = images.shape
        assert H == W == 224, f"Expected 224x224 input, got {H}x{W}"
        
        Vfeatures_list = []
        
        # 分别处理每个时间步的图像
        for t in range(T):
            current_images = images[:, t, :, :, :]  # [B, C, H, W]
            
            with torch.no_grad() if hasattr(self, 'freeze_backbone') else torch.enable_grad():
                outputs = dinov2_model(current_images)
                # 获取patch embeddings
                patch_embeddings = outputs.last_hidden_state  # [B, N+1, 768]
                # 移除CLS token
                base_features = patch_embeddings[:, 1:, :]  # [B, N, 768]
            
            # 特征维度适配
            base_features = feature_adapter(base_features)  # [B, N, 256]
            
            # 添加视角嵌入
            base_features = base_features + view_embed.expand(B, base_features.shape[1], -1)
            
            # 对于224x224输入，DINOv2的patch数量是固定的
            # DINOv2-base使用14x14的patch size，所以224x224图像会产生16x16=256个patches
            num_patches = base_features.shape[1]  # 应该是256
            patch_h = patch_w = int(math.sqrt(num_patches))  # 应该是16
            
            # 确保patch数量正确
            if patch_h * patch_w != num_patches:
                patch_h = patch_w = 16  # 强制设置为16x16
                base_features = base_features[:, :patch_h*patch_w, :]
            
            # 重塑为空间特征图
            spatial_features = base_features.view(B, patch_h, patch_w, 256).permute(0, 3, 1, 2)  # [B, 256, 16, 16]
            
            # 生成多尺度特征
            multi_scale_features = []
            
            # 4个尺度的特征
            scales = [1, 2, 4, 8]  # 下采样倍数
            
            for i, scale in enumerate(scales):
                if scale == 1:
                    feat = spatial_features  # [B, 256, 16, 16]
                else:
                    feat = F.avg_pool2d(spatial_features, kernel_size=scale, stride=scale)
                
                # 展平并添加位置编码
                feat_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, 256]
                feat_flat = feat_flat + self.level_embed[i].view(1, 1, -1)
                multi_scale_features.append(feat_flat)
            
            # 拼接所有尺度的特征
            Vfeatures = torch.cat(multi_scale_features, dim=1)  # [B, \sum H_i*W_i, 256]
            Vfeatures_list.append(Vfeatures)
        
        return Vfeatures_list
    
    def forward(self, images):
        """
        Args:
            images[0]: [B, T, C, H, W] 第一个视角的输入图像，H=W=224
            images[1]: [B, T, C, H, W] 第二个视角的输入图像，H=W=224
        Returns:
            Vfeatures_list: List[[B, \sum H_i*W_i, 256]] 每个时间步的融合多尺度特征
            spatial_shapes: [n_levels, 2] 每个尺度的空间形状
            level_start_index: [n_levels] 每个尺度的起始索引
            valid_ratios: [B, n_levels, 2] 有效区域比例
        """
        B, T, C, H, W = images[0].shape
        assert H == W == 224, f"Expected 224x224 input, got {H}x{W}"
        assert images[0].shape == images[1].shape, "Both views must have the same shape"
        
        # 编码两个视角
        Vfeatures_list_view1 = self.encode_single_view(
            images[0], self.dinov2_view1, self.feature_adapter_view1, self.view1_embed
        )
        Vfeatures_list_view2 = self.encode_single_view(
            images[1], self.dinov2_view2, self.feature_adapter_view2, self.view2_embed
        )
        
        # 融合两个视角的特征
        fused_Vfeatures_list = []
        for t in range(T):
            view1_features = Vfeatures_list_view1[t]  # [B, N, 256]
            view2_features = Vfeatures_list_view2[t]  # [B, N, 256]
            
            # 使用交叉注意力融合两个视角
            fused_features_1, _ = self.view_fusion(
                query=view1_features,
                key=view2_features,
                value=view2_features
            )
            
            fused_features_2, _ = self.view_fusion(
                query=view2_features,
                key=view1_features,
                value=view1_features
            )
            
            # 平均融合两个方向的注意力结果
            fused_features = (fused_features_1 + fused_features_2) / 2
            
            # 残差连接和投影
            fused_features = self.layer_norm(fused_features)
            fused_features = self.fusion_projection(fused_features)
            
            fused_Vfeatures_list.append(fused_features)
        
        # 计算spatial_shapes和level_start_index（基于224x224输入）
        spatial_shapes = []
        level_start_index = []
        scales = [1, 2, 4, 8]
        
        # 对于224x224输入，基础patch尺寸是16x16
        base_patch_h = base_patch_w = 16
        
        start_idx = 0
        for i, scale in enumerate(scales):
            if scale == 1:
                h, w = base_patch_h, base_patch_w  # 16, 16
            else:
                h, w = max(1, base_patch_h // scale), max(1, base_patch_w // scale)
            
            spatial_shapes.append([h, w])
            level_start_index.append(start_idx)
            start_idx += h * w
        
        # 转换为tensor格式
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=images[0].device)
        level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=images[0].device)
        
        # 有效区域比例（假设全部有效）
        valid_ratios = torch.ones(B, len(scales), 2, device=images[0].device)
        
        return fused_Vfeatures_list, spatial_shapes, level_start_index, valid_ratios

class DINOv2VisualEncoder(nn.Module):
    """基于DINOv2的视觉编码器 - 适配224x224输入"""
    def __init__(self, model_name="facebook/dinov2-base", freeze_backbone=False, image_size=(224, 224)):
        super().__init__()
        # 加载预训练的DINOv2模型
        self.dinov2 = Dinov2Model.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        # 特征维度适配
        self.feature_dim = self.dinov2.config.hidden_size  # 768 for base model
        self.feature_adapter = nn.Linear(self.feature_dim, 256)
        
        # 多尺度特征投影
        self.level_embed = nn.Parameter(torch.Tensor(4, 256))
        nn.init.normal_(self.level_embed)
        
    def forward(self, images):
        """
        Args:
            images: [B, T, C, H, W] 输入图像，H=W=224
        Returns:
            Vfeatures_list: List[[B, \sum H_i*W_i, 256]] 每个时间步的多尺度特征
            spatial_shapes: [n_levels, 2] 每个尺度的空间形状
            level_start_index: [n_levels] 每个尺度的起始索引
            valid_ratios: [B, n_levels, 2] 有效区域比例
        """
        B, T, C, H, W = images.shape
        assert H == W == 224, f"Expected 224x224 input, got {H}x{W}"
        
        Vfeatures_list = []
        spatial_shapes = []  # 存储[h, w]对
        level_start_index = []
        
        # 分别处理每个时间步的图像
        for t in range(T):
            current_images = images[:, t, :, :, :]  # [B, C, H, W]
            
            with torch.no_grad() if hasattr(self, 'freeze_backbone') else torch.enable_grad():
                outputs = self.dinov2(current_images)
                # 获取patch embeddings
                patch_embeddings = outputs.last_hidden_state  # [B, N+1, 768]
                # 移除CLS token
                base_features = patch_embeddings[:, 1:, :]  # [B, N, 768]
            
            # 特征维度适配
            base_features = self.feature_adapter(base_features)  # [B, N, 256]
            
            # 对于224x224输入，patch数量是固定的256个(16x16)
            num_patches = base_features.shape[1]
            patch_h = patch_w = 16  # 固定为16x16
            
            # 确保patch数量正确
            if num_patches != 256:
                base_features = base_features[:, :256, :]
            
            # 重塑为空间特征图
            spatial_features = base_features.view(B, patch_h, patch_w, 256).permute(0, 3, 1, 2)  # [B, 256, 16, 16]
            
            # 生成多尺度特征
            multi_scale_features = []
            
            # 4个尺度的特征
            scales = [1, 2, 4, 8]  # 下采样倍数
            start_idx = 0
            
            for i, scale in enumerate(scales):
                if scale == 1:
                    feat = spatial_features  # [B, 256, 16, 16]
                else:
                    feat = F.avg_pool2d(spatial_features, kernel_size=scale, stride=scale)
                
                h, w = feat.shape[2], feat.shape[3]
                if t == 0:  # 只在第一个时间步记录spatial_shapes和level_start_index
                    spatial_shapes.append([h, w])  # 添加[h, w]对
                    level_start_index.append(start_idx)
                start_idx += h * w
                
                # 展平并添加位置编码
                feat_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, 256]
                feat_flat = feat_flat + self.level_embed[i].view(1, 1, -1)
                multi_scale_features.append(feat_flat)
            
            # 拼接所有尺度的特征
            Vfeatures = torch.cat(multi_scale_features, dim=1)  # [B, \sum H_i*W_i, 256]
            Vfeatures_list.append(Vfeatures)
        
        # 转换为正确的2D tensor格式
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=images.device)  # [n_levels, 2]
        level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=images.device)  # [n_levels]
        
        # 有效区域比例（假设全部有效）
        valid_ratios = torch.ones(B, len(scales), 2, device=images.device)
        
        return Vfeatures_list, spatial_shapes, level_start_index, valid_ratios

class TemporalSpatialFusion(nn.Module):
    """时空特征融合模块"""
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 时序自注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 空间交叉注意力
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 层归一化
        self.temporal_norm = nn.LayerNorm(feature_dim)
        self.spatial_norm = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim),
            # nn.Dropout(0.1)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, Vfeatures_list):
        """
        Args:
            Vfeatures_list: List[[B, N, 256]] 包含T个时间步的视觉特征列表
        Returns:
            fused_features: [B, N, 256] 融合后的特征
        """
        # 将列表中的特征拼接成时序维度
        # Vfeatures_list: [tensor1[B, N, 256], tensor2[B, N, 256]]
        # 转换为: [B, T, N, 256]
        temporal_features = torch.stack(Vfeatures_list, dim=1)  # [B, T, N, 256]
        B, T, N, D = temporal_features.shape
        
        # 重塑为 [B*N, T, 256] 以便进行时序注意力
        temporal_features = temporal_features.permute(0, 2, 1, 3).contiguous()  # [B, N, T, 256]
        temporal_features = temporal_features.view(B * N, T, D)  # [B*N, T, 256]
        
        # 时序自注意力
        temporal_out, _ = self.temporal_attention(temporal_features, temporal_features, temporal_features)
        temporal_features = self.temporal_norm(temporal_features + temporal_out)
        
        # 取最后一个时间步的特征或者平均池化
        # 这里我们取最后一个时间步（当前时刻）
        current_features = temporal_features[:, -1, :]  # [B*N, 256]
        current_features = current_features.view(B, N, D)  # [B, N, 256]
        
        # 空间自注意力
        spatial_out, _ = self.spatial_attention(current_features, current_features, current_features)
        current_features = self.spatial_norm(current_features + spatial_out)
        
        # 前馈网络
        ffn_out = self.ffn(current_features)
        fused_features = self.ffn_norm(current_features + ffn_out)
        
        return fused_features

class PixelTransformerDecoder(nn.Module):
    """基于Transformer的像素级解码器 - 使用纯Transformer上采样到224x224"""
    def __init__(self, feature_dim=256, num_classes=10, image_size=(224, 224)):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.image_size = image_size
        
        # 明确定义：image_size = (height, width) = (224, 224)
        self.image_height = image_size[0]   # 224
        self.image_width = image_size[1]    # 224
        
        # 使用少量可学习的分割查询而不是每像素查询
        self.num_seg_queries = 100
        self.seg_queries = nn.Parameter(torch.randn(self.num_seg_queries, feature_dim))
        self.seg_pos_embed = nn.Parameter(torch.randn(self.num_seg_queries, feature_dim))
        
        # Transformer解码器层（减少层数）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # 基于Transformer的上采样网络
        # 定义多个分辨率级别的像素查询
        self.target_pixels = self.image_height * self.image_width  # 224 * 224 = 50176
        
        # 可学习的像素位置嵌入 (224x224)
        self.pixel_pos_embed = nn.Parameter(
            torch.randn(1, self.target_pixels, feature_dim)
        )
        
        # 像素查询生成器
        self.pixel_query_generator = nn.Linear(feature_dim, self.target_pixels)
        
        # 创建上采样阶段并分别处理模块和参数
        self.stage1, self.pixel_queries_56 = self._create_upsampling_stage(feature_dim, 56 * 56)
        self.stage2, self.pixel_queries_112 = self._create_upsampling_stage(feature_dim, 112 * 112)
        self.stage3, self.pixel_queries_224 = self._create_upsampling_stage(feature_dim, self.target_pixels)
        
        # 将阶段模块放入ModuleList
        self.upsampling_transformer = nn.ModuleList([self.stage1, self.stage2, self.stage3])
        
        # 注册参数
        self.register_parameter('pixel_queries_56', self.pixel_queries_56)
        self.register_parameter('pixel_queries_112', self.pixel_queries_112)
        self.register_parameter('pixel_queries_224', self.pixel_queries_224)
        
        # 最终分类头
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def _create_upsampling_stage(self, feature_dim, target_size):
        """创建单个上采样阶段"""
        stage = nn.ModuleDict({
            'cross_attention': nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            ),
            'self_attention': nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            ),
            'norm1': nn.LayerNorm(feature_dim),
            'norm2': nn.LayerNorm(feature_dim),
            'norm3': nn.LayerNorm(feature_dim),
            'ffn': nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim)
            )
        })
        
        # 单独创建pixel_queries参数
        pixel_queries = nn.Parameter(torch.randn(1, target_size, feature_dim))
        
        return stage, pixel_queries
        
    def forward(self, seg_features, visual_memory):
        """
        Args:
            seg_features: [B, num_seg_queries, feature_dim] 分割特征
            visual_memory: [B, N, feature_dim] 视觉记忆特征
        Returns:
            pixel_logits: [B, num_classes, H, W] 像素级分类结果 (H=W=224)
        """
        batch_size = seg_features.shape[0]
        device = seg_features.device
        
        # 获取分割查询
        seg_queries = self.seg_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        seg_queries_with_pos = seg_queries + self.seg_pos_embed.unsqueeze(0)
        
        # 构建记忆：分割特征 + 视觉特征
        memory = torch.cat([seg_features, visual_memory], dim=1)
        
        # Transformer解码
        decoded_features = self.transformer_decoder(
            tgt=seg_queries_with_pos,
            memory=memory
        )
        
        # 特征融合
        fused_features = self.feature_fusion(decoded_features)
        
        # 使用Transformer进行多阶段上采样
        current_features = fused_features
        
        # 获取像素查询参数列表
        pixel_queries_list = [self.pixel_queries_56, self.pixel_queries_112, self.pixel_queries_224]
        
        for i, stage in enumerate(self.upsampling_transformer):
            # 获取当前阶段的像素查询
            pixel_queries = pixel_queries_list[i].repeat(batch_size, 1, 1)
            
            # 交叉注意力：像素查询关注特征
            cross_attn_out, _ = stage['cross_attention'](
                query=pixel_queries,
                key=current_features,
                value=current_features
            )
            pixel_queries = stage['norm1'](pixel_queries + cross_attn_out)
            
            # 自注意力：像素间交互
            self_attn_out, _ = stage['self_attention'](
                query=pixel_queries,
                key=pixel_queries,
                value=pixel_queries
            )
            pixel_queries = stage['norm2'](pixel_queries + self_attn_out)
            
            # 前馈网络
            ffn_out = stage['ffn'](pixel_queries)
            current_features = stage['norm3'](pixel_queries + ffn_out)
        
        # 最终分类
        pixel_logits_flat = self.classification_head(current_features)  # [B, 50176, num_classes]
        
        # 重塑为图像格式
        pixel_logits = pixel_logits_flat.transpose(1, 2)  # [B, num_classes, 50176]
        pixel_logits = pixel_logits.view(
            batch_size, self.num_classes, self.image_height, self.image_width
        )  # [B, num_classes, 224, 224]
        
        return pixel_logits

class UnifiedTransformerHead(nn.Module):
    """统一的Transformer头，整合语义分割、目标检测和点预测"""
    def __init__(self, 
                 feature_dim=256, 
                 num_seg_queries=256,
                 num_det_queries=256, 
                 num_point_queries=3,
                 num_seg_classes=10,
                 num_det_classes=10,
                 num_layers=6,
                 n_levels=4,
                 action_dim=14,
                 image_size=(224, 224)):
        super().__init__()
        
        self.num_seg_queries = num_seg_queries
        self.num_det_queries = num_det_queries
        self.num_point_queries = num_point_queries
        self.num_seg_classes = num_seg_classes
        self.num_det_classes = num_det_classes
        self.image_size = image_size
        
        # 统一的查询嵌入
        total_queries = num_seg_queries + num_det_queries + num_point_queries
        self.unified_query_embed = nn.Embedding(total_queries, feature_dim)
        self.unified_query_pos = nn.Embedding(total_queries, feature_dim)
        
        # 任务特定的查询类型嵌入
        self.task_type_embed = nn.Embedding(3, feature_dim)  # 3种任务类型
        
        # 统一的Transformer解码器
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=feature_dim,
            n_levels=n_levels,
            n_heads=8,
            n_points=4
        )
        self.unified_decoder = DeformableTransformerDecoder(
            decoder_layer, num_layers, return_intermediate=True
        )
        
        # 跨任务注意力模块
        self.cross_task_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        self.cross_task_norm = nn.LayerNorm(feature_dim)
        
        # 任务特定的输出头
        # 1. 语义分割特征头
        self.seg_feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 2. 基于Transformer的像素级分割解码器
        self.pixel_transformer_decoder = PixelTransformerDecoder(
            feature_dim=feature_dim,
            num_classes=num_seg_classes,
            image_size=image_size
        )
        
        # 3. 目标检测头
        self.det_class_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_det_classes + 1)
        )
        
        self.det_bbox_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 4)
        )
        
        # 4. 点预测头
        self.point_coord_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim)  # 抓住物体时候机械臂的状态
        )
        
        # 参考点初始化
        self.reference_points = nn.Linear(feature_dim, 4)
        
        # 查询间交互的权重
        self.seg_to_det_proj = nn.Linear(feature_dim, feature_dim)
        self.det_to_seg_proj = nn.Linear(feature_dim, feature_dim)
        self.seg_to_point_proj = nn.Linear(feature_dim, feature_dim)
        self.det_to_point_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, Vfeatures, spatial_shapes, level_start_index, valid_ratios, padding_mask=None):
        """
        Args:
            Vfeatures: [B, \sum H_i*W_i, 256] 多尺度特征
            spatial_shapes: [n_levels, 2] 每个尺度的空间形状
            level_start_index: [n_levels] 每个尺度的起始索引
            valid_ratios: [B, n_levels, 2] 有效区域比例
            padding_mask: [B, \sum H_i*W_i] 填充掩码
        Returns:
            unified_outputs: 包含所有任务输出的字典
        """
        B = Vfeatures.shape[0]
        
        # 构建统一的查询嵌入
        unified_queries = self.unified_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        unified_query_pos = self.unified_query_pos.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # 添加任务类型嵌入
        task_embeds = []
        # 分割任务嵌入
        seg_task_embed = self.task_type_embed.weight[0].unsqueeze(0).repeat(B, self.num_seg_queries, 1)
        task_embeds.append(seg_task_embed)
        # 检测任务嵌入
        det_task_embed = self.task_type_embed.weight[1].unsqueeze(0).repeat(B, self.num_det_queries, 1)
        task_embeds.append(det_task_embed)
        # 点预测任务嵌入
        point_task_embed = self.task_type_embed.weight[2].unsqueeze(0).repeat(B, self.num_point_queries, 1)
        task_embeds.append(point_task_embed)
        
        task_type_embeds = torch.cat(task_embeds, dim=1)  # [B, total_queries, 256]
        
        # 融合查询嵌入和任务类型嵌入
        enhanced_queries = unified_queries + unified_query_pos + task_type_embeds
        
        # 初始化参考点
        reference_points = self.reference_points(enhanced_queries).sigmoid()
        
        # 统一的可变形解码器
        hs, inter_references = self.unified_decoder(
            tgt=enhanced_queries,
            reference_points=reference_points,
            src=Vfeatures,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=unified_query_pos,
            src_padding_mask=padding_mask
        )
        
        # 使用最后一层的输出
        unified_features = hs[-1]  # [B, total_queries, 256]
        
        # 分离不同任务的特征
        seg_start = 0
        seg_end = self.num_seg_queries
        det_start = seg_end
        det_end = det_start + self.num_det_queries
        point_start = det_end
        point_end = point_start + self.num_point_queries
        
        seg_features = unified_features[:, seg_start:seg_end, :]  # [B, num_seg_queries, 256]
        det_features = unified_features[:, det_start:det_end, :]   # [B, num_det_queries, 256]
        point_features = unified_features[:, point_start:point_end, :]  # [B, num_point_queries, 256]
        
        # 跨任务信息交互
        # 分割 <-> 检测
        seg_enhanced, _ = self.cross_task_attention(
            query=seg_features,
            key=det_features,
            value=det_features
        )
        seg_features = self.cross_task_norm(seg_features + seg_enhanced)
        
        det_enhanced, _ = self.cross_task_attention(
            query=det_features,
            key=seg_features,
            value=seg_features
        )
        det_features = self.cross_task_norm(det_features + det_enhanced)
        
        # 点预测与分割、检测的交互
        seg_to_point = self.seg_to_point_proj(seg_features.mean(dim=1, keepdim=True))  # [B, 1, 256]
        det_to_point = self.det_to_point_proj(det_features.mean(dim=1, keepdim=True))  # [B, 1, 256]
        
        point_enhanced = point_features + seg_to_point + det_to_point
        
        # 任务特定的输出预测
        # 1. 语义分割输出
        seg_query_features = self.seg_feature_head(seg_features)
        
        # 使用Transformer解码器进行像素级分割预测
        seg_logits = self.pixel_transformer_decoder(
            seg_features=seg_query_features,
            visual_memory=Vfeatures
        )  # [B, num_classes, H, W] - 直接输出到目标尺寸
        
        # 2. 目标检测输出
        det_class_logits = self.det_class_head(det_features)  # [B, num_det_queries, num_classes+1]
        det_bbox_coords = self.det_bbox_head(det_features).sigmoid()  # [B, num_det_queries, 4] 应用sigmoid
        
        # 3. 点预测输出
        point_coords = self.point_coord_head(point_enhanced)  # [B, num_points, 7]
        
        return {
            'seg_features': seg_query_features,
            'seg_logits': seg_logits,
            'det_features': det_features,
            'class_logits': det_class_logits,
            'bbox_coords': det_bbox_coords,
            'point_features': point_enhanced,
            'point_coords': point_coords,
            'unified_features': unified_features  # 用于轨迹预测的统一特征
        }

class RobotUniADModel224(nn.Module):
    """适配224x224输入的机器人统一模型"""
    def __init__(self, 
                 num_seg_classes=10,
                 num_det_classes=10, 
                 num_seg_queries=256,
                 num_det_queries=256,
                 num_points=3,
                 action_dim=7,
                 seq_len=30,
                 feature_dim=256,
                 image_size=(224, 224),
                 use_markov_regularizer=True,
                 markov_weight=0.1,
                 use_triple_view=False,
                 use_dual_view=True):
        super().__init__()
        
        # 1. 视觉编码器
        if use_triple_view:
            self.visual_encoder = TripleViewDINOv2VisualEncoder(image_size=image_size)
            self.num_views = 3
        elif use_dual_view:
            self.visual_encoder = DualViewDINOv2VisualEncoder(image_size=image_size)
            self.num_views = 2
        else:
            self.visual_encoder = DINOv2VisualEncoder(image_size=image_size)
            self.num_views = 1
        
        # 2. 时空特征融合
        self.temporal_spatial_fusion = TemporalSpatialFusion(feature_dim)
        
        # 3. 统一的Transformer头
        self.unified_transformer = UnifiedTransformerHead(
            feature_dim=feature_dim,
            num_seg_queries=num_seg_queries,
            num_det_queries=num_det_queries,
            num_point_queries=num_points,
            num_seg_classes=num_seg_classes,
            num_det_classes=num_det_classes,
            action_dim = action_dim,
            image_size=image_size
        )
        
        # 4. 动作轨迹模型
        self.dit_action_head = DITActionHead(
            action_dim=action_dim,
            seq_len=seq_len,
            hidden_size=512,
            depth=6,
            num_heads=8,
            feature_dim=feature_dim,
            num_classes=1000,
            learn_sigma=True,
            dropout_prob=0.1
        )
        
        # 扩散调度器
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # 5. DETR风格的损失计算组件
        self.matcher = HungarianMatcher()
        self.criterion = DETRLossComputer(
            num_classes=num_det_classes,
            matcher=self.matcher,
            weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2},
            eos_coef=0.1,
            losses=['labels', 'boxes']
        )
        
    def forward(self, images, input_actions=None, target_actions=None, det_targets=None):
        """
        Args:
            images: 输入图像序列
                - 单视角: [B, T, C, H, W] H=W=224
                - 双视角: ([B, T, C, H, W], [B, T, C, H, W]) H=W=224
                - 三视角: ([B, T, C, H, W], [B, T, C, H, W], [B, T, C, H, W]) H=W=224
            input_actions: [B, action_dim] 当前动作状态
            target_actions: [B, seq_len, action_dim] 目标动作序列（训练时使用）
            det_targets: List[Dict] 检测目标标注（训练时使用）
        Returns:
            outputs: 包含所有任务输出的字典
        """
        # 验证输入尺寸
        if self.num_views == 1:
            B, T, C, H, W = images.shape
            assert H == W == 224, f"Expected 224x224 input, got {H}x{W}"
        else:
            B, T, C, H, W = images[0].shape
            assert H == W == 224, f"Expected 224x224 input, got {H}x{W}"
            # 验证所有视角的尺寸一致
            for i in range(self.num_views):
                assert images[i].shape == (B, T, C, H, W), f"View {i} shape mismatch: expected {(B, T, C, H, W)}, got {images[i].shape}"
        
        # 1. 视觉编码
        Vfeatures_list, spatial_shapes, level_start_index, valid_ratios = self.visual_encoder(images)
        
        # 2. 时空特征融合
        Vfeatures = self.temporal_spatial_fusion(Vfeatures_list)  # [B, N, 256]
        
        # 3. 统一Transformer处理所有任务
        unified_outputs = self.unified_transformer(
            Vfeatures, spatial_shapes, level_start_index, valid_ratios
        )
        
        # 4. DIT动作生成模型
        if self.num_views == 1:
            device = images.device
        else:
            device = images[0].device
        
        if self.training and target_actions is not None:
            # 训练模式：扩散训练
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=device)
            
            # 添加噪声到目标动作
            noise = torch.randn_like(target_actions)
            noisy_actions = self.scheduler.add_noise(target_actions, noise, timesteps)
            
            # 使用当前状态作为条件
            if input_actions is not None:
                current_state = input_actions
            else:
                current_state = torch.zeros(B, self.dit_action_head.action_dim, device=device)
            
            # DIT前向传播预测噪声
            predicted_noise = self.dit_action_head(
                noisy_actions,
                timesteps,
                current_state,
                unified_outputs['seg_features'],
                unified_outputs['det_features'],
                Vfeatures,
                unified_outputs['point_features']
            )
            
            trajectory_output = predicted_noise
            # 保存训练时的噪声用于损失计算
            self._training_noise = noise
            self._training_timesteps = timesteps
            
        else:
            # 推理模式：扩散采样
            if input_actions is not None:
                current_state = input_actions
            else:
                current_state = torch.zeros(B, self.dit_action_head.action_dim, device=device)
            
            # 从纯噪声开始采样
            trajectory_output = torch.randn(B, self.dit_action_head.seq_len, 
                                          self.dit_action_head.action_dim, device=device)
            
            # 扩散采样过程
            for t in reversed(range(self.scheduler.num_train_timesteps)):
                timesteps = torch.full((B,), t, device=device, dtype=torch.long)
                
                # 预测噪声
                predicted_noise = self.dit_action_head(
                    trajectory_output,
                    timesteps,
                    current_state,
                    unified_outputs['seg_features'],
                    unified_outputs['det_features'],
                    Vfeatures,
                    unified_outputs['point_features']
                )
                
                if self.dit_action_head.learn_sigma:
                    # 分离噪声预测和方差预测
                    predicted_noise, predicted_variance = predicted_noise.chunk(2, dim=-1)
                
                # 去噪一步
                trajectory_output = self.scheduler.step(predicted_noise, t, trajectory_output)
        
        # 整合所有输出
        outputs = {
            'seg_logits': unified_outputs['seg_logits'],  # [B, num_classes, 224, 224]
            'class_logits': unified_outputs['class_logits'],
            'bbox_coords': unified_outputs['bbox_coords'],
            'point_coords': unified_outputs['point_coords'],
            'trajectory_actions': trajectory_output,  # [B, seq_len, action_dim]
            'det_features': unified_outputs['det_features']
        }
        return outputs
    
    def compute_loss(self, outputs, targets):
        """计算多任务损失"""
        losses = {}
        
        # 1. 语义分割损失（像素级）
        if 'seg_masks' in targets:
            # 确保目标掩码也是224x224
            seg_masks = targets['seg_masks']
            if seg_masks.shape[-2:] != (224, 224):
                seg_masks = F.interpolate(
                    seg_masks.unsqueeze(1).float(), 
                    size=(224, 224), 
                    mode='nearest'
                ).squeeze(1).long()
            
            seg_loss = F.cross_entropy(
                outputs['seg_logits'], 
                seg_masks
            )
            losses['seg_loss'] = seg_loss
        
        # 2. 目标检测损失（DETR风格）
        if 'det_targets' in targets:
            det_outputs = {
                'pred_logits': outputs['class_logits'],
                'pred_boxes': outputs['bbox_coords']
            }
            det_losses = self.criterion(det_outputs, targets['det_targets'])
            losses['det_bbox_loss'] = det_losses['loss_bbox']
            losses['det_ce_loss'] = det_losses['loss_ce']
        
        # 3. 抓取点损失
        if 'point_coords' in targets:
            point_loss = F.l1_loss(
                outputs['point_coords'], 
                targets['point_coords']
            )
            losses['point_loss'] = point_loss
        
        # 4. DIT扩散生成损失
        if 'slave_actions' in targets:
            if self.training and hasattr(self, '_training_noise'):
                # 训练模式：计算噪声预测损失
                predicted_noise = outputs['trajectory_actions']
                if predicted_noise.shape[-1] == self.dit_action_head.action_dim * 2:
                    predicted_noise = predicted_noise[..., :self.dit_action_head.action_dim]
                
                trajectory_loss = F.mse_loss(
                    predicted_noise,
                    self._training_noise
                )
            else:
                # 推理模式：计算动作预测损失
                predicted_actions = outputs['trajectory_actions']
                if predicted_actions.shape[-1] == self.dit_action_head.action_dim * 2:
                    predicted_actions = predicted_actions[..., :self.dit_action_head.action_dim]
                
                trajectory_loss = F.mse_loss(
                    predicted_actions, 
                    targets['master_actions']
                )
            losses['trajectory_loss'] = trajectory_loss
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses

# 示例使用
if __name__ == "__main__":
    # 创建适配224x224的统一Transformer模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试三视角模型
    print("=== 测试三视角模型 ===")
    model_triple = RobotUniADModel224(
        num_seg_classes=1,
        num_det_classes=3,
        num_seg_queries=5,
        num_det_queries=5,
        num_points=5,
        action_dim=14,
        seq_len=30,
        image_size=(224, 224),
        use_triple_view=True,
        use_dual_view=False
    )
    model_triple = model_triple.to(device)
    print(f"三视角模型参数数量: {sum(p.numel() for p in model_triple.parameters() if p.requires_grad):,}")
    
    # 测试三视角前向传播
    batch_size = 1
    timesteps = 1
    images1 = torch.randn(batch_size, timesteps, 3, 224, 224).to(device)
    images2 = torch.randn(batch_size, timesteps, 3, 224, 224).to(device)
    images3 = torch.randn(batch_size, timesteps, 3, 224, 224).to(device)
    images_triple = (images1, images2, images3)
    
    model_triple.eval()
    with torch.no_grad():
        outputs_triple = model_triple(images_triple)
        
        print("\n三视角模型输出形状:")
        for key, value in outputs_triple.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
    
    # 测试双视角模型
    print("\n=== 测试双视角模型 ===")
    model_dual = RobotUniADModel224(
        num_seg_classes=1,
        num_det_classes=3,
        num_seg_queries=50,
        num_det_queries=50,
        num_points=5,
        action_dim=14,
        seq_len=30,
        image_size=(224, 224),
        use_triple_view=False,
        use_dual_view=True
    )
    model_dual = model_dual.to(device)
    print(f"双视角模型参数数量: {sum(p.numel() for p in model_dual.parameters() if p.requires_grad):,}")
    
    # 测试双视角前向传播
    images_dual = (images1, images2)
    
    model_dual.eval()
    with torch.no_grad():
        outputs_dual = model_dual(images_dual)
        
        print("\n双视角模型输出形状:")
        for key, value in outputs_dual.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
    
    # 测试单视角模型
    print("\n=== 测试单视角模型 ===")
    model_single = RobotUniADModel224(
        num_seg_classes=1,
        num_det_classes=3,
        num_seg_queries=50,
        num_det_queries=50,
        num_points=5,
        action_dim=14,
        seq_len=30,
        image_size=(224, 224),
        use_triple_view=False,
        use_dual_view=False
    )
    model_single = model_single.to(device)
    print(f"单视角模型参数数量: {sum(p.numel() for p in model_single.parameters() if p.requires_grad):,}")
    
    # 测试单视角前向传播
    model_single.eval()
    with torch.no_grad():
        outputs_single = model_single(images1)
        
        print("\n单视角模型输出形状:")
        for key, value in outputs_single.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
    
    # 验证分割输出尺寸
    print(f"\n输入图像尺寸: {images1.shape[2:]}")
    print(f"三视角分割输出尺寸: {outputs_triple['seg_logits'].shape[2:]}")
    print(f"双视角分割输出尺寸: {outputs_dual['seg_logits'].shape[2:]}")
    print(f"单视角分割输出尺寸: {outputs_single['seg_logits'].shape[2:]}")
    print(f"尺寸匹配: {outputs_triple['seg_logits'].shape[2:] == images1.shape[2:]}")
    
    # 验证特征图尺寸
    print(f"\n预期的多尺度特征尺寸:")
    print(f"尺度1 (16x16): {16*16} patches")
    print(f"尺度2 (8x8): {8*8} patches")
    print(f"尺度3 (4x4): {4*4} patches")
    print(f"尺度4 (2x2): {2*2} patches")
    print(f"总计: {16*16 + 8*8 + 4*4 + 2*2} patches")
    
    print("\n=== 模型修改完成 ===")
    print("现在支持三种模式:")
    print("1. 单视角模式: use_triple_view=False, use_dual_view=False")
    print("2. 双视角模式: use_triple_view=False, use_dual_view=True")
    print("3. 三视角模式: use_triple_view=True, use_dual_view=False")
