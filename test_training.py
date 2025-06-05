import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from VLAModel import RobotUniADModel
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_random_data(batch_size=4, device='cuda'):
    """
    生成随机训练数据
    """
    # 输入图像: [B, T, C, H, W]
    images = torch.randn(batch_size, 2, 3, 640, 480).to(device)
    
    # 输入动作序列: [B, seq_len, action_dim] - 用于自回归模型的特征编码
    input_actions = torch.randn(batch_size, 30, 7).to(device)
    
    # 目标动作序列: [B, seq_len, action_dim]
    target_actions = torch.randn(batch_size, 30, 7).to(device)
    
    # 语义分割掩码: [B, H, W]
    seg_masks = torch.randint(0, 1, (batch_size, 480, 640)).to(device)
    
    # 检测目标（DETR格式）
    det_targets = []
    for i in range(batch_size):
        num_objects = np.random.randint(1, 6)  # 每张图1-5个目标
        target = {
            'labels': torch.randint(0, 10, (num_objects,)).to(device),
            'boxes': torch.rand(num_objects, 4).to(device)  # [x_center, y_center, width, height] 归一化坐标
        }
        det_targets.append(target)
    
    # 抓取点坐标: [B, num_points, 7]
    point_coords = torch.rand(batch_size, 3, 7).to(device)  # 3个抓取点
    
    targets = {
        'seg_masks': seg_masks,
        'det_targets': det_targets,
        'point_coords': point_coords,
        'actions': target_actions
    }
    
    return images, input_actions, target_actions, targets

def test_model_training():
    """
    测试模型训练过程
    """
    print("开始VLA模型训练验证...")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = RobotUniADModel(
        num_seg_classes=1,
        num_det_classes=10,
        num_seg_queries=50,
        num_det_queries=50,
        num_points=3,
        action_dim=7,
        seq_len=30,
        image_size=(640, 480)
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 训练循环
    model.train()
    num_epochs = 1
    num_batches = 10
    
    loss_history = []
    
    print("\n开始训练循环...")
    for epoch in range(num_epochs):
        epoch_losses = []
        
        with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx in pbar:
                # 生成随机数据
                images, input_actions, target_actions, targets = generate_random_data(batch_size=2, device=device)
                
                # 前向传播 - 使用新的自回归模型接口
                try:
                    outputs = model(images, input_actions, target_actions)
                    
                    # 计算损失
                    losses = model.compute_loss(outputs, targets)
                    total_loss = losses['total_loss']
                    
                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # 记录损失
                    epoch_losses.append(total_loss.item())
                    loss_history.append(total_loss.item())
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'Seg': f'{losses.get("seg_loss", 0):.4f}',
                        'Det': f'{losses.get("loss_ce", 0):.4f}',
                        'Point': f'{losses.get("point_loss", 0):.4f}',
                        'Traj': f'{losses.get("trajectory_loss", 0):.4f}'
                    })
                    
                except Exception as e:
                    print(f"\n训练过程中出现错误: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    # 测试推理模式
    print("\n测试推理模式...")
    model.eval()
    with torch.no_grad():
        try:
            test_images = torch.randn(1, 2, 3, 640, 480).to(device)
            test_input_actions = torch.randn(1, 30, 7).to(device)
            inference_outputs = model(test_images, test_input_actions)
            
            print("推理输出形状:")
            for key, value in inference_outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
            
            print("✅ 推理模式测试成功!")
            
        except Exception as e:
            print(f"❌ 推理模式测试失败: {e}")
            return False
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('VLA模型训练损失曲线')
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.savefig('c:\\DiskD\\trae_doc\\VLA\\training_loss_curve.png')
    plt.show()
    
    print(f"\n✅ 模型训练验证完成!")
    print(f"最终损失: {loss_history[-1]:.4f}")
    print(f"损失曲线已保存到: c:\\DiskD\\trae_doc\\VLA\\training_loss_curve.png")
    
    return True

def test_model_components():
    """
    测试模型各个组件
    """
    print("\n测试模型组件...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RobotUniADModel(
        num_seg_classes=10,
        num_det_classes=10,
        num_seg_queries=50,
        num_det_queries=50,
        num_points=3,
        action_dim=7,
        seq_len=30,
        image_size=(640, 480)
    ).to(device)
    
    # 测试视觉编码器
    print("1. 测试视觉编码器...")
    test_images = torch.randn(2, 2, 3, 640, 480).to(device)
    try:
        Vfeatures_list, spatial_shapes, level_start_index, valid_ratios = model.visual_encoder(test_images)
        print(f"   ✅ 视觉特征形状: {[f.shape for f in Vfeatures_list]}")
    except Exception as e:
        print(f"   ❌ 视觉编码器错误: {e}")
        return False
    
    # 测试时空融合
    print("2. 测试时空融合...")
    try:
        Vfeatures = model.temporal_spatial_fusion(Vfeatures_list)
        print(f"   ✅ 融合特征形状: {Vfeatures.shape}")
    except Exception as e:
        print(f"   ❌ 时空融合错误: {e}")
        return False
    
    # 测试统一Transformer
    print("3. 测试统一Transformer...")
    try:
        unified_outputs = model.unified_transformer(
            Vfeatures, spatial_shapes, level_start_index, valid_ratios
        )
        print(f"   ✅ 统一输出键: {list(unified_outputs.keys())}")
    except Exception as e:
        print(f"   ❌ 统一Transformer错误: {e}")
        return False
    
    # 测试自回归轨迹生成
    print("4. 测试自回归轨迹生成...")
    try:
        test_input_actions = torch.randn(2, 30, 7).to(device)
        test_target_actions = torch.randn(2, 30, 7).to(device)
        
        trajectory_output = model.trajectory_autoregressive(
            test_input_actions,
            unified_outputs['seg_features'],
            unified_outputs['det_features'],
            Vfeatures,
            unified_outputs['point_features'],
            target_actions=test_target_actions
        )
        print(f"   ✅ 轨迹输出形状: {trajectory_output.shape}")
    except Exception as e:
        print(f"   ❌ 自回归轨迹生成错误: {e}")
        return False
    
    # 测试自回归推理模式
    print("5. 测试自回归推理模式...")
    try:
        model.eval()
        with torch.no_grad():
            inference_output = model.trajectory_autoregressive.generate(
                test_input_actions,
                unified_outputs['seg_features'],
                unified_outputs['det_features'],
                Vfeatures,
                unified_outputs['point_features']
            )
            print(f"   ✅ 推理输出形状: {inference_output.shape}")
    except Exception as e:
        print(f"   ❌ 自回归推理模式错误: {e}")
        return False
    
    print("✅ 所有组件测试通过!")
    return True

def test_autoregressive_generation():
    """
    专门测试自回归生成功能
    """
    print("\n测试自回归生成功能...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RobotUniADModel(
        num_seg_classes=10,
        num_det_classes=10,
        num_seg_queries=50,
        num_det_queries=50,
        num_points=3,
        action_dim=7,
        seq_len=30,
        image_size=(640, 480)
    ).to(device)
    
    model.eval()
    
    # 生成测试数据
    test_images = torch.randn(1, 2, 3, 640, 480).to(device)
    test_input_actions = torch.randn(1, 30, 7).to(device)
    
    print("1. 测试完整推理流程...")
    try:
        with torch.no_grad():
            outputs = model(test_images, test_input_actions)
            trajectory_actions = outputs['trajectory_actions']
            print(f"   ✅ 生成的动作序列形状: {trajectory_actions.shape}")
            print(f"   ✅ 动作值范围: [{trajectory_actions.min().item():.3f}, {trajectory_actions.max().item():.3f}]")
    except Exception as e:
        print(f"   ❌ 完整推理流程错误: {e}")
        return False
    
    print("2. 测试批量生成...")
    try:
        batch_images = torch.randn(4, 2, 3, 640, 480).to(device)
        batch_input_actions = torch.randn(4, 30, 7).to(device)
        
        with torch.no_grad():
            batch_outputs = model(batch_images, batch_input_actions)
            batch_trajectory = batch_outputs['trajectory_actions']
            print(f"   ✅ 批量生成形状: {batch_trajectory.shape}")
    except Exception as e:
        print(f"   ❌ 批量生成错误: {e}")
        return False
    
    print("✅ 自回归生成功能测试通过!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("VLA自回归模型训练验证脚本")
    print("=" * 60)
    
    # 测试模型组件
    if test_model_components():
        # 测试自回归生成
        if test_autoregressive_generation():
            # 测试完整训练流程
            success = test_model_training()
            
            if success:
                print("\n🎉 VLA自回归模型训练验证成功! 模型可以正常训练。")
            else:
                print("\n❌ VLA模型训练验证失败，请检查错误信息。")
        else:
            print("\n❌ 自回归生成功能测试失败，请检查模型定义。")
    else:
        print("\n❌ 模型组件测试失败，请检查模型定义。")
