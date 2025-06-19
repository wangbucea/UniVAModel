import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import os
from VLAModel import RobotUniADModel
from vla_dataset_loader import VLADataset, create_vla_dataloader
from torchvision import transforms

class SingleSampleValidator:
    """单样本验证器 - 验证模型各个输出头的正确性"""
    
    def __init__(self, model_path, data_root="c:/DiskD/trae_doc/VLA", device='cuda'):
        self.device = device
        self.data_root = data_root
        self.model_path = model_path
        
        # 初始化模型
        self.model = RobotUniADModel(
            vision_encoder_type='dual_view',
            num_classes=2,  # 背景+前景
            num_detection_classes=3,  # bottle, brush, cube
            action_dim=7,
            max_trajectory_length=30
        )
        
        # 加载模型权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"模型已加载: {model_path}")
        else:
            print(f"警告: 模型文件不存在 {model_path}，使用随机初始化的模型")
        
        self.model.to(device)
        self.model.eval()
        
        # 初始化数据集
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = VLADataset(
            data_root=data_root,
            sequence_length=5,
            action_sequence_length=30,
            transform=transform
        )
        
        # 类别映射
        self.class_names = ['bottle', 'brush', 'cube']
        self.seg_class_names = ['background', 'foreground']
        
    def validate_sample(self, sample_idx=0, save_results=True):
        """验证指定样本的模型输出"""
        print(f"\n=== 验证样本 {sample_idx} ===")
        
        # 获取样本数据
        sample = self.dataset[sample_idx]
        print(f"样本ID: {sample['sample_id']}")
        print(f"起始帧: {sample['start_frame']}")
        
        # 准备输入数据
        batch = self._prepare_batch(sample)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(
                chest_images=batch['chest_images'],
                head_images=batch['head_images'],
                current_state=batch['current_state'],
                mode='inference'
            )
        
        # 验证各个输出头
        results = {
            'sample_info': {
                'sample_id': sample['sample_id'],
                'start_frame': sample['start_frame']
            }
        }
        
        # 1. 验证语义分割
        seg_results = self._validate_segmentation(outputs, sample, batch)
        results['segmentation'] = seg_results
        
        # 2. 验证目标检测
        det_results = self._validate_detection(outputs, sample, batch)
        results['detection'] = det_results
        
        # 3. 验证关键帧动作
        key_frame_results = self._validate_key_frame_actions(outputs, sample, batch)
        results['key_frame_actions'] = key_frame_results
        
        # 4. 验证生成动作
        action_results = self._validate_generated_actions(outputs, sample, batch)
        results['generated_actions'] = action_results
        
        # 可视化结果
        if save_results:
            self._visualize_results(sample, outputs, results)
        
        # 打印验证结果
        self._print_validation_summary(results)
        
        return results
    
    def _prepare_batch(self, sample):
        """准备批次数据"""
        batch = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).to(self.device)
            else:
                batch[key] = value
        return batch
    
    def _validate_segmentation(self, outputs, sample, batch):
        """验证语义分割输出"""
        print("\n--- 验证语义分割 ---")
        
        # 获取分割预测
        seg_logits = outputs.get('seg_logits')
        if seg_logits is None:
            print("警告: 模型未输出分割结果")
            return {'error': '无分割输出'}
        
        # 获取预测结果
        seg_pred = torch.argmax(seg_logits, dim=1).cpu().numpy()[0]  # [H, W]
        seg_probs = torch.softmax(seg_logits, dim=1).cpu().numpy()[0]  # [C, H, W]
        
        # 获取真实标签
        gt_mask = sample['segmentation_mask'].cpu().numpy()  # [H, W]
        
        # 计算指标
        pixel_acc = (seg_pred == gt_mask).mean()
        
        # 计算每个类别的IoU
        ious = []
        for class_id in range(len(self.seg_class_names)):
            pred_mask = (seg_pred == class_id)
            gt_class_mask = (gt_mask == class_id)
            
            intersection = (pred_mask & gt_class_mask).sum()
            union = (pred_mask | gt_class_mask).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if intersection == 0 else 0.0
            
            ious.append(iou)
            print(f"  {self.seg_class_names[class_id]} IoU: {iou:.4f}")
        
        mean_iou = np.mean(ious)
        
        print(f"  像素准确率: {pixel_acc:.4f}")
        print(f"  平均IoU: {mean_iou:.4f}")
        
        return {
            'pixel_accuracy': float(pixel_acc),
            'mean_iou': float(mean_iou),
            'class_ious': [float(iou) for iou in ious],
            'prediction_shape': seg_pred.shape,
            'gt_shape': gt_mask.shape,
            'num_classes': len(self.seg_class_names)
        }
    
    def _validate_detection(self, outputs, sample, batch):
        """验证目标检测输出"""
        print("\n--- 验证目标检测 ---")
        
        # 获取检测预测
        detection_outputs = outputs.get('detection_outputs')
        if detection_outputs is None:
            print("警告: 模型未输出检测结果")
            return {'error': '无检测输出'}
        
        # 解析检测输出
        pred_boxes = detection_outputs.get('pred_boxes', torch.tensor([]))
        pred_logits = detection_outputs.get('pred_logits', torch.tensor([]))
        
        if pred_boxes.numel() == 0 or pred_logits.numel() == 0:
            print("警告: 检测输出为空")
            return {'error': '检测输出为空'}
        
        # 转换为numpy
        pred_boxes = pred_boxes.cpu().numpy()[0]  # [num_queries, 4]
        pred_logits = pred_logits.cpu().numpy()[0]  # [num_queries, num_classes]
        pred_probs = torch.softmax(torch.tensor(pred_logits), dim=-1).numpy()
        
        # 获取真实标签
        gt_boxes = sample['detection_boxes'].cpu().numpy()  # [num_gt, 4]
        gt_labels = sample['detection_labels'].cpu().numpy()  # [num_gt]
        
        # 过滤高置信度预测
        confidence_threshold = 0.5
        max_probs = pred_probs.max(axis=1)
        pred_labels = pred_probs.argmax(axis=1)
        
        valid_preds = max_probs > confidence_threshold
        filtered_boxes = pred_boxes[valid_preds]
        filtered_labels = pred_labels[valid_preds]
        filtered_probs = max_probs[valid_preds]
        
        print(f"  真实目标数量: {len(gt_boxes)}")
        print(f"  预测目标数量: {len(filtered_boxes)}")
        print(f"  置信度阈值: {confidence_threshold}")
        
        # 打印预测结果
        for i, (box, label, prob) in enumerate(zip(filtered_boxes, filtered_labels, filtered_probs)):
            class_name = self.class_names[label] if label < len(self.class_names) else f"class_{label}"
            print(f"  预测 {i+1}: {class_name} ({prob:.3f}) - 框: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
        
        # 打印真实标签
        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            class_name = self.class_names[label] if label < len(self.class_names) else f"class_{label}"
            print(f"  真实 {i+1}: {class_name} - 框: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
        
        return {
            'num_predictions': len(filtered_boxes),
            'num_ground_truth': len(gt_boxes),
            'confidence_threshold': confidence_threshold,
            'predicted_boxes': filtered_boxes.tolist(),
            'predicted_labels': filtered_labels.tolist(),
            'predicted_probs': filtered_probs.tolist(),
            'ground_truth_boxes': gt_boxes.tolist(),
            'ground_truth_labels': gt_labels.tolist()
        }
    
    def _validate_key_frame_actions(self, outputs, sample, batch):
        """验证关键帧动作预测"""
        print("\n--- 验证关键帧动作 ---")
        
        # 获取关键帧预测
        key_frame_outputs = outputs.get('key_frame_predictions')
        if key_frame_outputs is None:
            print("警告: 模型未输出关键帧预测")
            return {'error': '无关键帧输出'}
        
        # 转换为numpy
        pred_key_frames = key_frame_outputs.cpu().numpy()[0]  # [5, 7]
        gt_key_frames = sample['key_frame_labels'].cpu().numpy()  # [5, 7]
        
        # 反归一化动作（如果需要）
        if hasattr(self.dataset, 'denormalize_action'):
            pred_key_frames_denorm = np.array([self.dataset.denormalize_action(action) for action in pred_key_frames])
            gt_key_frames_denorm = np.array([self.dataset.denormalize_action(action) for action in gt_key_frames])
        else:
            pred_key_frames_denorm = pred_key_frames
            gt_key_frames_denorm = gt_key_frames
        
        # 计算误差
        mse_error = np.mean((pred_key_frames - gt_key_frames) ** 2)
        mae_error = np.mean(np.abs(pred_key_frames - gt_key_frames))
        
        print(f"  关键帧数量: {len(pred_key_frames)}")
        print(f"  MSE误差: {mse_error:.6f}")
        print(f"  MAE误差: {mae_error:.6f}")
        
        # 打印每个关键帧的对比
        for i in range(min(5, len(pred_key_frames))):
            print(f"  关键帧 {i+1}:")
            print(f"    预测: [{', '.join([f'{x:.3f}' for x in pred_key_frames_denorm[i]])}]")
            print(f"    真实: [{', '.join([f'{x:.3f}' for x in gt_key_frames_denorm[i]])}]")
            frame_error = np.mean(np.abs(pred_key_frames[i] - gt_key_frames[i]))
            print(f"    误差: {frame_error:.6f}")
        
        return {
            'mse_error': float(mse_error),
            'mae_error': float(mae_error),
            'num_key_frames': len(pred_key_frames),
            'predicted_actions': pred_key_frames.tolist(),
            'ground_truth_actions': gt_key_frames.tolist(),
            'predicted_actions_denorm': pred_key_frames_denorm.tolist(),
            'ground_truth_actions_denorm': gt_key_frames_denorm.tolist()
        }
    
    def _validate_generated_actions(self, outputs, sample, batch):
        """验证生成的动作序列"""
        print("\n--- 验证生成动作 ---")
        
        # 获取生成的动作序列
        generated_actions = outputs.get('predicted_actions')
        if generated_actions is None:
            print("警告: 模型未输出生成动作")
            return {'error': '无生成动作输出'}
        
        # 转换为numpy
        pred_actions = generated_actions.cpu().numpy()[0]  # [seq_len, 7]
        gt_actions = sample['master_actions'].cpu().numpy()  # [seq_len, 7]
        
        # 确保序列长度一致
        min_len = min(len(pred_actions), len(gt_actions))
        pred_actions = pred_actions[:min_len]
        gt_actions = gt_actions[:min_len]
        
        # 反归一化动作（如果需要）
        if hasattr(self.dataset, 'denormalize_action'):
            pred_actions_denorm = np.array([self.dataset.denormalize_action(action) for action in pred_actions])
            gt_actions_denorm = np.array([self.dataset.denormalize_action(action) for action in gt_actions])
        else:
            pred_actions_denorm = pred_actions
            gt_actions_denorm = gt_actions
        
        # 计算误差
        mse_error = np.mean((pred_actions - gt_actions) ** 2)
        mae_error = np.mean(np.abs(pred_actions - gt_actions))
        
        # 计算轨迹相似度
        trajectory_similarity = self._calculate_trajectory_similarity(pred_actions, gt_actions)
        
        print(f"  动作序列长度: {len(pred_actions)}")
        print(f"  MSE误差: {mse_error:.6f}")
        print(f"  MAE误差: {mae_error:.6f}")
        print(f"  轨迹相似度: {trajectory_similarity:.4f}")
        
        # 打印前几个时间步的对比
        for i in range(min(5, len(pred_actions))):
            print(f"  时间步 {i+1}:")
            print(f"    预测: [{', '.join([f'{x:.3f}' for x in pred_actions_denorm[i]])}]")
            print(f"    真实: [{', '.join([f'{x:.3f}' for x in gt_actions_denorm[i]])}]")
            step_error = np.mean(np.abs(pred_actions[i] - gt_actions[i]))
            print(f"    误差: {step_error:.6f}")
        
        return {
            'mse_error': float(mse_error),
            'mae_error': float(mae_error),
            'trajectory_similarity': float(trajectory_similarity),
            'sequence_length': len(pred_actions),
            'predicted_actions': pred_actions.tolist(),
            'ground_truth_actions': gt_actions.tolist(),
            'predicted_actions_denorm': pred_actions_denorm.tolist(),
            'ground_truth_actions_denorm': gt_actions_denorm.tolist()
        }
    
    def _calculate_trajectory_similarity(self, pred_traj, gt_traj):
        """计算轨迹相似度"""
        # 使用余弦相似度计算轨迹相似度
        pred_flat = pred_traj.flatten()
        gt_flat = gt_traj.flatten()
        
        # 计算余弦相似度
        dot_product = np.dot(pred_flat, gt_flat)
        norm_pred = np.linalg.norm(pred_flat)
        norm_gt = np.linalg.norm(gt_flat)
        
        if norm_pred == 0 or norm_gt == 0:
            return 0.0
        
        similarity = dot_product / (norm_pred * norm_gt)
        return similarity
    
    def _visualize_results(self, sample, outputs, results):
        """可视化验证结果"""
        print("\n--- 生成可视化结果 ---")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"样本 {sample['sample_id']} 验证结果", fontsize=16)
        
        # 1. 显示原始图像（Chest摄像头中间帧）
        ax = axes[0, 0]
        chest_img = sample['chest_images'][2]  # 中间帧
        # 反归一化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        chest_img_denorm = chest_img * std + mean
        chest_img_denorm = torch.clamp(chest_img_denorm, 0, 1)
        ax.imshow(chest_img_denorm.permute(1, 2, 0))
        ax.set_title('原始图像 (Chest)')
        ax.axis('off')
        
        # 2. 显示语义分割结果
        if 'segmentation' in results and 'error' not in results['segmentation']:
            ax = axes[0, 1]
            seg_logits = outputs.get('seg_logits')
            if seg_logits is not None:
                seg_pred = torch.argmax(seg_logits, dim=1).cpu().numpy()[0]
                ax.imshow(seg_pred, cmap='tab10')
                ax.set_title(f'分割预测 (mIoU: {results["segmentation"]["mean_iou"]:.3f})')
            ax.axis('off')
        
        # 3. 显示目标检测结果
        if 'detection' in results and 'error' not in results['detection']:
            ax = axes[0, 2]
            ax.imshow(chest_img_denorm.permute(1, 2, 0))
            
            # 绘制预测框
            det_results = results['detection']
            for box, label, prob in zip(det_results['predicted_boxes'], 
                                      det_results['predicted_labels'], 
                                      det_results['predicted_probs']):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                       edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                class_name = self.class_names[label] if label < len(self.class_names) else f"class_{label}"
                ax.text(x1, y1-5, f'{class_name}: {prob:.2f}', 
                       color='red', fontsize=8, weight='bold')
            
            # 绘制真实框
            for box, label in zip(det_results['ground_truth_boxes'], 
                                det_results['ground_truth_labels']):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                       edgecolor='green', facecolor='none')
                ax.add_patch(rect)
                class_name = self.class_names[label] if label < len(self.class_names) else f"class_{label}"
                ax.text(x1, y2+5, f'GT: {class_name}', 
                       color='green', fontsize=8, weight='bold')
            
            ax.set_title(f'目标检测 (预测: {det_results["num_predictions"]}, 真实: {det_results["num_ground_truth"]})')
            ax.axis('off')
        
        # 4. 显示关键帧动作对比
        if 'key_frame_actions' in results and 'error' not in results['key_frame_actions']:
            ax = axes[1, 0]
            key_results = results['key_frame_actions']
            pred_actions = np.array(key_results['predicted_actions'])
            gt_actions = np.array(key_results['ground_truth_actions'])
            
            x = np.arange(len(pred_actions))
            width = 0.35
            
            # 只显示前3个关节的对比
            for joint_idx in range(min(3, pred_actions.shape[1])):
                ax.bar(x - width/2, pred_actions[:, joint_idx], width, 
                      label=f'预测关节{joint_idx+1}', alpha=0.7)
                ax.bar(x + width/2, gt_actions[:, joint_idx], width, 
                      label=f'真实关节{joint_idx+1}', alpha=0.7)
            
            ax.set_xlabel('关键帧')
            ax.set_ylabel('关节角度')
            ax.set_title(f'关键帧动作对比 (MAE: {key_results["mae_error"]:.4f})')
            ax.legend()
        
        # 5. 显示生成动作轨迹
        if 'generated_actions' in results and 'error' not in results['generated_actions']:
            ax = axes[1, 1]
            action_results = results['generated_actions']
            pred_traj = np.array(action_results['predicted_actions'])
            gt_traj = np.array(action_results['ground_truth_actions'])
            
            # 只显示前3个关节的轨迹
            for joint_idx in range(min(3, pred_traj.shape[1])):
                ax.plot(pred_traj[:, joint_idx], '--', 
                       label=f'预测关节{joint_idx+1}', alpha=0.7)
                ax.plot(gt_traj[:, joint_idx], '-', 
                       label=f'真实关节{joint_idx+1}', alpha=0.7)
            
            ax.set_xlabel('时间步')
            ax.set_ylabel('关节角度')
            ax.set_title(f'动作轨迹对比 (相似度: {action_results["trajectory_similarity"]:.4f})')
            ax.legend()
        
        # 6. 显示误差统计
        ax = axes[1, 2]
        metrics = []
        values = []
        
        if 'segmentation' in results and 'error' not in results['segmentation']:
            metrics.append('分割mIoU')
            values.append(results['segmentation']['mean_iou'])
        
        if 'key_frame_actions' in results and 'error' not in results['key_frame_actions']:
            metrics.append('关键帧MAE')
            values.append(results['key_frame_actions']['mae_error'])
        
        if 'generated_actions' in results and 'error' not in results['generated_actions']:
            metrics.append('轨迹相似度')
            values.append(results['generated_actions']['trajectory_similarity'])
        
        if metrics:
            ax.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax.set_title('性能指标汇总')
            ax.set_ylabel('数值')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = f"validation_sample_{sample['sample_id']}_frame_{sample['start_frame']}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
        plt.show()
    
    def _print_validation_summary(self, results):
        """打印验证结果摘要"""
        print("\n" + "="*60)
        print("验证结果摘要")
        print("="*60)
        
        sample_info = results['sample_info']
        print(f"样本ID: {sample_info['sample_id']}")
        print(f"起始帧: {sample_info['start_frame']}")
        
        # 语义分割结果
        if 'segmentation' in results:
            seg_results = results['segmentation']
            if 'error' not in seg_results:
                print(f"\n语义分割:")
                print(f"  ✓ 像素准确率: {seg_results['pixel_accuracy']:.4f}")
                print(f"  ✓ 平均IoU: {seg_results['mean_iou']:.4f}")
            else:
                print(f"\n语义分割: ✗ {seg_results['error']}")
        
        # 目标检测结果
        if 'detection' in results:
            det_results = results['detection']
            if 'error' not in det_results:
                print(f"\n目标检测:")
                print(f"  ✓ 预测目标数: {det_results['num_predictions']}")
                print(f"  ✓ 真实目标数: {det_results['num_ground_truth']}")
            else:
                print(f"\n目标检测: ✗ {det_results['error']}")
        
        # 关键帧动作结果
        if 'key_frame_actions' in results:
            key_results = results['key_frame_actions']
            if 'error' not in key_results:
                print(f"\n关键帧动作:")
                print(f"  ✓ MAE误差: {key_results['mae_error']:.6f}")
                print(f"  ✓ MSE误差: {key_results['mse_error']:.6f}")
            else:
                print(f"\n关键帧动作: ✗ {key_results['error']}")
        
        # 生成动作结果
        if 'generated_actions' in results:
            action_results = results['generated_actions']
            if 'error' not in action_results:
                print(f"\n生成动作:")
                print(f"  ✓ MAE误差: {action_results['mae_error']:.6f}")
                print(f"  ✓ 轨迹相似度: {action_results['trajectory_similarity']:.4f}")
            else:
                print(f"\n生成动作: ✗ {action_results['error']}")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    # 配置参数
    model_path = "path/to/your/trained_model.pth"  # 替换为您的模型路径
    data_root = "c:/DiskD/trae_doc/VLA"  # 数据根目录
    sample_idx = 0  # 要验证的样本索引
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在 {model_path}")
        print("将使用随机初始化的模型进行演示")
    
    # 初始化验证器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    validator = SingleSampleValidator(
        model_path=model_path,
        data_root=data_root,
        device=device
    )
    
    # 验证样本
    try:
        results = validator.validate_sample(sample_idx=sample_idx, save_results=True)
        
        # 保存验证结果
        with open(f"validation_results_sample_{sample_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n验证结果已保存到: validation_results_sample_{sample_idx}.json")
        
    except Exception as e:
        print(f"验证过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
