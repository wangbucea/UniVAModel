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
