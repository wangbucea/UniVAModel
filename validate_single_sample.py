import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

# 导入模型和数据加载器
from VLAModel import RobotUniADModel
from vla_dataset_loader import VLADataset, collate_fn

class VLASingleSampleValidator:
    def __init__(self, model_path, data_root, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_root = data_root
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 创建数据集
        self.dataset = self._create_dataset()
        
        # 类别映射
        self.idx_to_class = {0: 'bottle', 1: 'brush', 2: 'cube'}
        
    def _load_model(self, model_path):
        """加载训练好的模型"""
        model = RobotUniADModel(
            num_seg_classes=2,  # 背景 + 前景
            num_det_classes=3,  # bottle, brush, cube
            num_seg_queries=50,
            num_det_queries=50,
            num_points=5,
            action_dim=7,
            seq_len=30,
            feature_dim=256,
            image_size=(480, 640),  # 注意：这里是(H, W)
            use_dual_view=True
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"警告：模型文件 {model_path} 不存在，使用随机初始化的模型")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _create_dataset(self):
        """创建数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = VLADataset(
            data_root=self.data_root,
            transform=transform,
            stride=1
        )
        return dataset
    
    def validate_sample(self, sample_idx=0):
        """验证指定样本"""
        print(f"\n=== 验证样本 {sample_idx} ===")
        
        # 获取单个样本
        sample = self.dataset[sample_idx]
        
        # 创建批次（添加batch维度）
        batch = {
            'chest_images': sample['chest_images'].unsqueeze(0).to(self.device),
            'head_images': sample['head_images'].unsqueeze(0).to(self.device),
            'current_state': sample['current_state'].unsqueeze(0).to(self.device),
            'slave_actions': sample['slave_actions'].unsqueeze(0).to(self.device),
            'master_actions': sample['master_actions'].unsqueeze(0).to(self.device),
            'detection_boxes': [sample['detection_boxes']],
            'detection_labels': [sample['detection_labels']],
            'segmentation_mask': sample['segmentation_mask'].unsqueeze(0).to(self.device),
            'key_frame_labels': sample['key_frame_labels'].unsqueeze(0).to(self.device)
        }
        
        print(f"样本ID: {sample['sample_id']}")
        print(f"起始帧: {sample['start_frame']}")
        
        # 模型推理
        with torch.no_grad():
            # 准备输入
            images = [batch['chest_images'], batch['head_images']]
            input_actions = batch['current_state']
            
            # 前向传播
            outputs = self.model(
                images=images,
                input_actions=input_actions,
                target_actions=None  # 推理模式
            )
        
        # 验证各个输出
        results = {
            'sample_info': {
                'sample_id': sample['sample_id'],
                'start_frame': sample['start_frame']
            },
            'segmentation': self._validate_segmentation(outputs, batch),
            'detection': self._validate_detection(outputs, batch),
            'keyframe_actions': self._validate_keyframe_actions(outputs, batch),
            'generated_actions': self._validate_generated_actions(outputs, batch)
        }
        
        return results
    
    def _validate_segmentation(self, outputs, batch):
        """验证语义分割"""
        print("\n--- 语义分割验证 ---")
        
        # 获取预测结果
        seg_logits = outputs['seg_logits']  # [B, C, H, W]
        seg_pred = torch.argmax(seg_logits, dim=1)  # [B, H, W]
        
        # 获取真实标签 - 修正维度顺序
        seg_gt = batch['segmentation_mask']  # [B, W, H] -> 需要转置为 [B, H, W]
        seg_gt = seg_gt.transpose(1, 2)  # 转置最后两个维度
        
        print(f"预测分割形状: {seg_pred.shape}")
        print(f"真实分割形状: {seg_gt.shape}")
        
        # 计算准确率
        correct = (seg_pred == seg_gt).float()
        accuracy = correct.mean().item()
        
        # 计算IoU
        intersection = ((seg_pred == 1) & (seg_gt == 1)).float().sum()
        union = ((seg_pred == 1) | (seg_gt == 1)).float().sum()
        iou = (intersection / (union + 1e-8)).item()
        
        print(f"分割准确率: {accuracy:.4f}")
        print(f"前景IoU: {iou:.4f}")
        
        return {
            'accuracy': accuracy,
            'iou': iou,
            'pred_shape': list(seg_pred.shape),
            'gt_shape': list(seg_gt.shape)
        }
    
    def _validate_detection(self, outputs, batch):
        """验证目标检测"""
        print("\n--- 目标检测验证 ---")
        
        # 获取预测结果
        class_logits = outputs['class_logits']  # [B, num_queries, num_classes]
        bbox_coords = outputs['bbox_coords']    # [B, num_queries, 4]
        
        # 获取真实标签
        gt_boxes = batch['detection_boxes'][0]  # 第一个样本的检测框
        gt_labels = batch['detection_labels'][0]  # 第一个样本的标签
        
        print(f"预测类别logits形状: {class_logits.shape}")
        print(f"预测边界框形状: {bbox_coords.shape}")
        print(f"真实检测框数量: {len(gt_boxes)}")
        print(f"真实标签: {[self.idx_to_class[label.item()] for label in gt_labels]}")
        
        # 获取置信度最高的预测
        pred_probs = torch.softmax(class_logits[0], dim=-1)  # [num_queries, num_classes]
        pred_scores, pred_labels = torch.max(pred_probs, dim=-1)  # [num_queries]
        
        # 过滤低置信度预测（假设背景类是最后一个类别）
        valid_mask = pred_labels < (class_logits.shape[-1] - 1)  # 非背景类
        confidence_mask = pred_scores > 0.5  # 置信度阈值
        final_mask = valid_mask & confidence_mask
        
        if final_mask.sum() > 0:
            final_boxes = bbox_coords[0][final_mask]
            final_labels = pred_labels[final_mask]
            final_scores = pred_scores[final_mask]
            
            print(f"高置信度预测数量: {len(final_boxes)}")
            for i, (box, label, score) in enumerate(zip(final_boxes, final_labels, final_scores)):
                print(f"  预测 {i+1}: {self.idx_to_class[label.item()]} (置信度: {score:.3f})")
                print(f"    边界框: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
        else:
            print("没有高置信度的预测")
            final_boxes = torch.empty(0, 4)
            final_labels = torch.empty(0, dtype=torch.long)
            final_scores = torch.empty(0)
        
        return {
            'num_predictions': len(final_boxes),
            'num_gt': len(gt_boxes),
            'predictions': [
                {
                    'class': self.idx_to_class[label.item()],
                    'confidence': score.item(),
                    'bbox': box.tolist()
                }
                for box, label, score in zip(final_boxes, final_labels, final_scores)
            ],
            'ground_truth': [
                {
                    'class': self.idx_to_class[label.item()],
                    'bbox': box.tolist()
                }
                for box, label in zip(gt_boxes, gt_labels)
            ]
        }
    
    def _validate_keyframe_actions(self, outputs, batch):
        """验证关键帧动作预测"""
        print("\n--- 关键帧动作验证 ---")
        
        # 获取预测结果
        point_coords = outputs['point_coords']  # [B, num_points, action_dim]
        
        # 获取真实关键帧动作
        gt_keyframe_actions = batch['key_frame_labels']  # [B, 5, 7]
        
        print(f"预测关键帧动作形状: {point_coords.shape}")
        print(f"真实关键帧动作形状: {gt_keyframe_actions.shape}")
        
        # 计算L1损失
        l1_loss = torch.nn.functional.l1_loss(point_coords, gt_keyframe_actions)
        
        # 计算L2损失
        l2_loss = torch.nn.functional.mse_loss(point_coords, gt_keyframe_actions)
        
        print(f"关键帧动作L1损失: {l1_loss:.6f}")
        print(f"关键帧动作L2损失: {l2_loss:.6f}")
        
        # 反归一化以查看实际值（如果需要）
        if hasattr(self.dataset, 'denormalize_action'):
            pred_denorm = self.dataset.denormalize_action(point_coords[0].cpu().numpy())
            gt_denorm = self.dataset.denormalize_action(gt_keyframe_actions[0].cpu().numpy())
            
            print("\n关键帧动作对比（反归一化后）:")
            for i in range(min(5, len(pred_denorm))):
                print(f"  关键帧 {i+1}:")
                print(f"    预测: {pred_denorm[i]}")
                print(f"    真实: {gt_denorm[i]}")
        
        return {
            'l1_loss': l1_loss.item(),
            'l2_loss': l2_loss.item(),
            'pred_shape': list(point_coords.shape),
            'gt_shape': list(gt_keyframe_actions.shape)
        }
    
    def _validate_generated_actions(self, outputs, batch):
        """验证生成的动作序列"""
        print("\n--- 生成动作序列验证 ---")
        
        # 获取预测结果
        trajectory_actions = outputs['trajectory_actions']  # [B, seq_len, action_dim]
        
        # 获取真实动作序列
        gt_slave_actions = batch['slave_actions']  # [B, seq_len, 7]
        gt_master_actions = batch['master_actions']  # [B, seq_len, 7]
        
        print(f"生成动作序列形状: {trajectory_actions.shape}")
        print(f"真实slave动作形状: {gt_slave_actions.shape}")
        print(f"真实master动作形状: {gt_master_actions.shape}")
        
        # 处理DIT输出维度（如果learn_sigma=True，输出维度可能是action_dim*2）
        if trajectory_actions.shape[-1] == 14:  # action_dim * 2
            trajectory_actions = trajectory_actions[..., :7]  # 只取前7维
            print("检测到learn_sigma=True，只使用前7维进行评估")
        
        # 与slave actions比较（输入序列）
        slave_l1_loss = torch.nn.functional.l1_loss(trajectory_actions, gt_slave_actions)
        slave_l2_loss = torch.nn.functional.mse_loss(trajectory_actions, gt_slave_actions)
        
        # 与master actions比较（目标序列）
        master_l1_loss = torch.nn.functional.l1_loss(trajectory_actions, gt_master_actions)
        master_l2_loss = torch.nn.functional.mse_loss(trajectory_actions, gt_master_actions)
        
        print(f"与slave动作的L1损失: {slave_l1_loss:.6f}")
        print(f"与slave动作的L2损失: {slave_l2_loss:.6f}")
        print(f"与master动作的L1损失: {master_l1_loss:.6f}")
        print(f"与master动作的L2损失: {master_l2_loss:.6f}")
        
        # 计算动作序列的统计信息
        pred_mean = trajectory_actions.mean(dim=(0, 1))
        pred_std = trajectory_actions.std(dim=(0, 1))
        slave_mean = gt_slave_actions.mean(dim=(0, 1))
        slave_std = gt_slave_actions.std(dim=(0, 1))
        
        print("\n动作序列统计信息:")
        print(f"预测动作均值: {pred_mean.cpu().numpy()}")
        print(f"预测动作标准差: {pred_std.cpu().numpy()}")
        print(f"真实slave动作均值: {slave_mean.cpu().numpy()}")
        print(f"真实slave动作标准差: {slave_std.cpu().numpy()}")
        
        return {
            'slave_l1_loss': slave_l1_loss.item(),
            'slave_l2_loss': slave_l2_loss.item(),
            'master_l1_loss': master_l1_loss.item(),
            'master_l2_loss': master_l2_loss.item(),
            'pred_shape': list(trajectory_actions.shape),
            'pred_mean': pred_mean.cpu().numpy().tolist(),
            'pred_std': pred_std.cpu().numpy().tolist()
        }
    
    def save_validation_report(self, results, output_path):
        """保存验证报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n验证报告已保存到: {output_path}")
    
    def visualize_results(self, sample_idx=0, save_path=None):
        """可视化验证结果"""
        sample = self.dataset[sample_idx]
        
        # 创建图像网格
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'VLA模型验证结果 - 样本 {sample_idx}', fontsize=16)
        
        # 显示原始图像
        chest_img = sample['chest_images'][0]  # 第一帧
        head_img = sample['head_images'][0]    # 第一帧
        
        # 反归一化图像用于显示
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        chest_img_denorm = chest_img * std + mean
        head_img_denorm = head_img * std + mean
        
        axes[0, 0].imshow(chest_img_denorm.permute(1, 2, 0).clamp(0, 1))
        axes[0, 0].set_title('Chest Camera')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(head_img_denorm.permute(1, 2, 0).clamp(0, 1))
        axes[0, 1].set_title('Head Camera')
        axes[0, 1].axis('off')
        
        # 显示分割掩码
        seg_mask = sample['segmentation_mask'].numpy()
        axes[0, 2].imshow(seg_mask, cmap='gray')
        axes[0, 2].set_title('Segmentation Mask')
        axes[0, 2].axis('off')
        
        # 显示动作序列对比
        slave_actions = sample['slave_actions'].numpy()
        master_actions = sample['master_actions'].numpy()
        
        # 绘制前3个关节的动作轨迹
        for i in range(3):
            axes[1, i].plot(slave_actions[:, i], label='Slave Actions', alpha=0.7)
            axes[1, i].plot(master_actions[:, i], label='Master Actions', alpha=0.7)
            axes[1, i].set_title(f'Joint {i+1} Actions')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()

def main():
    """主函数"""
    # 配置参数
    model_path = "path/to/your/trained_model.pth"  # 替换为实际的模型路径
    data_root = "c:/DiskD/trae_doc/VLA"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 创建验证器
    validator = VLASingleSampleValidator(model_path, data_root, device)
    
    print(f"数据集大小: {len(validator.dataset)}")
    
    # 验证多个样本
    sample_indices = [0, 10, 20]  # 可以修改要验证的样本索引
    
    all_results = []
    
    for sample_idx in sample_indices:
        if sample_idx < len(validator.dataset):
            print(f"\n{'='*50}")
            print(f"验证样本 {sample_idx}")
            print(f"{'='*50}")
            
            # 执行验证
            results = validator.validate_sample(sample_idx)
            all_results.append(results)
            
            # 保存单个样本的验证报告
            report_path = f"validation_report_sample_{sample_idx}.json"
            validator.save_validation_report(results, report_path)
            
            # 可视化结果
            viz_path = f"validation_visualization_sample_{sample_idx}.png"
            validator.visualize_results(sample_idx, viz_path)
    
    # 保存综合报告
    comprehensive_report = {
        'summary': {
            'total_samples_validated': len(all_results),
            'device_used': device,
            'data_root': data_root
        },
        'results': all_results
    }
    
    validator.save_validation_report(comprehensive_report, "comprehensive_validation_report.json")
    
    print(f"\n{'='*50}")
    print("验证完成！")
    print(f"共验证了 {len(all_results)} 个样本")
    print("请查看生成的报告和可视化文件")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
