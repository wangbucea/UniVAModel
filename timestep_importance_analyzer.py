import h5py
import numpy as np
import matplotlib.pyplot as plt
# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import entropy
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict

class TimestepImportanceAnalyzer:
    def __init__(self, h5_file_path):
        """
        时间步重要性分析器
        
        Args:
            h5_file_path: HDF5数据文件路径
        """
        self.h5_file_path = h5_file_path
        self.trajectories = None  # Shape: [N, 30, 7]
        self.timestep_importance_scores = None  # Shape: [30]
        self.important_timesteps = None
        self.hierarchical_key_actions = {}  # 存储层次化关键动作结果
        
    def load_data(self):
        """
        从HDF5文件加载轨迹数据
        """
        print("正在加载轨迹数据...")
        trajectories = []
        
        with h5py.File(self.h5_file_path, 'r') as f:
            trajectory_group = f['trajectories']
            num_trajectories = len(trajectory_group.keys())
            
            for i in tqdm(range(num_trajectories)):
                if str(i) in trajectory_group:
                    traj_data = trajectory_group[str(i)][:]
                    if traj_data.shape == (30, 7):  # 确保数据格式正确
                        trajectories.append(traj_data)
        
        self.trajectories = np.array(trajectories)  # Shape: [N, 30, 7]
        print(f"成功加载 {len(self.trajectories)} 条轨迹")
        print(f"轨迹形状: {self.trajectories.shape}")
        
    def calculate_timestep_importance(self):
        """
        计算每个时间步的重要性分数
        """
        print("正在计算时间步重要性...")
        
        num_timesteps = self.trajectories.shape[1]  # 30
        num_joints = self.trajectories.shape[2]     # 7
        importance_scores = []
        
        for t in tqdm(range(num_timesteps)):
            # 获取所有轨迹在时间步t的动作
            timestep_actions = self.trajectories[:, t, :]  # Shape: [N, 7]
            
            # 1. 动作幅度重要性
            action_magnitude = np.mean(np.abs(timestep_actions))
            
            # 2. 动作变异性重要性（标准差）
            action_variance = np.mean(np.std(timestep_actions, axis=0))
            
            # 3. 信息熵重要性（离散化后计算熵）
            entropy_score = 0
            for joint in range(num_joints):
                joint_actions = timestep_actions[:, joint]
                # 将连续值离散化为bins
                hist, _ = np.histogram(joint_actions, bins=10, density=True)
                hist = hist + 1e-10  # 避免log(0)
                entropy_score += entropy(hist)
            entropy_score /= num_joints
            
            # 4. 相对于前一时间步的变化重要性
            if t > 0:
                prev_actions = self.trajectories[:, t-1, :]
                change_magnitude = np.mean(np.abs(timestep_actions - prev_actions))
            else:
                change_magnitude = action_magnitude  # 第一个时间步使用动作幅度
            
            # 5. 相对于后一时间步的变化重要性
            if t < num_timesteps - 1:
                next_actions = self.trajectories[:, t+1, :]
                future_change = np.mean(np.abs(next_actions - timestep_actions))
            else:
                future_change = action_magnitude  # 最后一个时间步使用动作幅度
            
            # 6. 聚类分离度重要性
            # 对当前时间步的动作进行聚类，看分离度如何
            if len(timestep_actions) > 10:  # 确保有足够的数据点
                try:
                    kmeans = KMeans(n_clusters=min(5, len(timestep_actions)//2), random_state=42)
                    labels = kmeans.fit_predict(timestep_actions)
                    # 计算轮廓系数作为分离度指标
                    from sklearn.metrics import silhouette_score
                    if len(set(labels)) > 1:
                        separation_score = silhouette_score(timestep_actions, labels)
                    else:
                        separation_score = 0
                except:
                    separation_score = 0
            else:
                separation_score = 0
            
            # 综合重要性分数
            importance = (
                0.2 * action_magnitude +
                0.2 * action_variance +
                0.2 * entropy_score +
                0.15 * change_magnitude +
                0.15 * future_change +
                0.1 * separation_score
            )
            
            importance_scores.append(importance)
        
        self.timestep_importance_scores = np.array(importance_scores)
        print("时间步重要性计算完成")
        
    def find_important_timesteps(self, num_important=6, method='top_k'):
        """
        找出最重要的时间步
        
        Args:
            num_important: 要找出的重要时间步数量
            method: 选择方法 ('top_k', 'threshold', 'adaptive')
            
        Returns:
            重要时间步的索引列表
        """
        if method == 'top_k':
            # 直接选择分数最高的k个时间步
            important_indices = np.argsort(self.timestep_importance_scores)[::-1][:num_important]
            
        elif method == 'threshold':
            # 选择分数超过阈值的时间步
            threshold = np.mean(self.timestep_importance_scores) + np.std(self.timestep_importance_scores)
            important_indices = np.where(self.timestep_importance_scores > threshold)[0]
            
        elif method == 'adaptive':
            # 自适应选择：结合分数和分布
            scores = self.timestep_importance_scores
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # 选择分数高于均值+0.5*标准差的时间步
            candidates = np.where(scores > mean_score + 0.5 * std_score)[0]
            
            # 如果候选太少，降低阈值
            if len(candidates) < num_important:
                candidates = np.argsort(scores)[::-1][:num_important]
            # 如果候选太多，提高阈值
            elif len(candidates) > num_important * 1.5:
                candidates = np.where(scores > mean_score + std_score)[0]
                if len(candidates) < num_important:
                    candidates = np.argsort(scores)[::-1][:num_important]
            
            important_indices = candidates
        
        # 排序以便输出
        important_indices = np.sort(important_indices)
        self.important_timesteps = important_indices
        
        print(f"\n发现的重要时间步 (使用{method}方法):")
        for i, timestep in enumerate(important_indices):
            score = self.timestep_importance_scores[timestep]
            print(f"  {i+1}. 时间步 {timestep} (重要性分数: {score:.4f})")
        
        return important_indices
    
    def extract_important_actions(self, important_timesteps=None):
        """
        提取所有轨迹在重要时间步的动作
        
        Args:
            important_timesteps: 重要时间步索引，如果为None则使用之前找到的
            
        Returns:
            重要动作数据，形状为 [N, len(important_timesteps), 7]
        """
        if important_timesteps is None:
            important_timesteps = self.important_timesteps
        
        if important_timesteps is None:
            raise ValueError("请先调用find_important_timesteps方法")
        
        # 提取重要时间步的动作
        important_actions = self.trajectories[:, important_timesteps, :]
        
        print(f"\n提取了 {len(self.trajectories)} 条轨迹在 {len(important_timesteps)} 个重要时间步的动作")
        print(f"重要动作数据形状: {important_actions.shape}")
        print(f"总共有 {len(self.trajectories) * len(important_timesteps)} 条重要动作序列")
        
        return important_actions
    
    def analyze_important_actions_patterns(self, important_timesteps=None):
        """
        分析重要时间步的动作模式
        """
        if important_timesteps is None:
            important_timesteps = self.important_timesteps
            
        important_actions = self.extract_important_actions(important_timesteps)
        
        print("\n=== 重要时间步动作模式分析 ===")
        
        for i, timestep in enumerate(important_timesteps):
            actions_at_timestep = important_actions[:, i, :]  # [N, 7]
            
            print(f"\n时间步 {timestep}:")
            print(f"  动作均值: {np.mean(actions_at_timestep, axis=0)}")
            print(f"  动作标准差: {np.std(actions_at_timestep, axis=0)}")
            print(f"  动作范围: [{np.min(actions_at_timestep, axis=0)}, {np.max(actions_at_timestep, axis=0)}]")
            
            # 分析每个关节的动作分布
            for joint in range(7):
                joint_actions = actions_at_timestep[:, joint]
                print(f"  关节{joint+1}: 均值={np.mean(joint_actions):.4f}, 标准差={np.std(joint_actions):.4f}")
    
    def visualize_timestep_importance(self, save_path=None):
        """
        可视化时间步重要性分析结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 时间步重要性分数曲线
        axes[0, 0].plot(range(30), self.timestep_importance_scores, 'b-', linewidth=2, marker='o')
        if self.important_timesteps is not None:
            axes[0, 0].scatter(self.important_timesteps, 
                             self.timestep_importance_scores[self.important_timesteps], 
                             color='red', s=100, zorder=5, label='重要时间步')
        axes[0, 0].set_title('时间步重要性分数')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('重要性分数')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. 重要性分数分布直方图
        axes[0, 1].hist(self.timestep_importance_scores, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(self.timestep_importance_scores), color='red', 
                          linestyle='--', label='均值')
        axes[0, 1].set_title('重要性分数分布')
        axes[0, 1].set_xlabel('重要性分数')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        
        # 3. 各时间步的动作幅度热力图
        action_magnitudes = np.mean(np.abs(self.trajectories), axis=0)  # [30, 7]
        im = axes[1, 0].imshow(action_magnitudes.T, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('各时间步各关节的平均动作幅度')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('关节')
        axes[1, 0].set_yticks(range(7))
        axes[1, 0].set_yticklabels([f'关节{i+1}' for i in range(7)])
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. 重要时间步的动作分布箱线图
        if self.important_timesteps is not None and len(self.important_timesteps) > 0:
            important_actions = self.extract_important_actions()
            # 重塑数据用于箱线图
            box_data = []
            labels = []
            for i, timestep in enumerate(self.important_timesteps):
                for joint in range(7):
                    box_data.append(important_actions[:, i, joint])
                    labels.append(f'T{timestep}J{joint+1}')
            
            # 只显示前几个以避免过于拥挤
            max_display = min(21, len(box_data))  # 最多显示21个
            axes[1, 1].boxplot(box_data[:max_display], labels=labels[:max_display])
            axes[1, 1].set_title('重要时间步各关节动作分布')
            axes[1, 1].set_xlabel('时间步-关节')
            axes[1, 1].set_ylabel('动作值')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def get_batch_important_actions(self, batch_indices=None, important_timesteps=None):
        """
        获取指定批次在重要时间步的动作
        
        Args:
            batch_indices: 批次索引列表，如果为None则返回所有批次
            important_timesteps: 重要时间步，如果为None则使用已找到的
            
        Returns:
            字典，包含每个批次的重要动作
        """
        if important_timesteps is None:
            important_timesteps = self.important_timesteps
            
        if batch_indices is None:
            batch_indices = list(range(len(self.trajectories)))
        
        batch_important_actions = {}
        
        for batch_idx in batch_indices:
            # 提取该批次在重要时间步的动作
            important_actions = self.trajectories[batch_idx, important_timesteps, :]
            batch_important_actions[batch_idx] = {
                'timesteps': important_timesteps,
                'actions': important_actions,
                'shape': important_actions.shape
            }
        
        print(f"\n为 {len(batch_indices)} 个批次提取了重要动作:")
        print(f"重要时间步: {important_timesteps}")
        print(f"每个批次的重要动作形状: {important_actions.shape}")
        
        return batch_important_actions
    
    def hierarchical_clustering_analysis(self, cluster_centers_list=[20, 10, 5, 3]):
        """
        层次化聚类分析
        
        Args:
            cluster_centers_list: 聚类中心数列表，从大到小
            
        Returns:
            层次化聚类结果字典
        """
        print(f"开始层次化聚类分析，聚类中心数序列: {cluster_centers_list}")
        
        # 准备数据：将所有轨迹的所有时间步展平
        # Shape: [N*30, 7] -> [N*30, 7]
        all_timestep_actions = self.trajectories.reshape(-1, 7)
        
        hierarchical_results = {}
        
        for level, n_clusters in enumerate(cluster_centers_list):
            print(f"\n--- 聚类层级 {level+1}: {n_clusters} 个聚类中心 ---")
            
            # 执行K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(all_timestep_actions)
            cluster_centers = kmeans.cluster_centers_
            
            # 分析每个聚类中的时间步分布
            cluster_timestep_distribution = defaultdict(list)
            
            # 将聚类标签重新映射到轨迹和时间步
            for traj_idx in range(len(self.trajectories)):
                for timestep in range(30):
                    global_idx = traj_idx * 30 + timestep
                    cluster_id = cluster_labels[global_idx]
                    cluster_timestep_distribution[cluster_id].append((traj_idx, timestep))
            
            # 计算每个聚类的时间步重要性
            cluster_importance = {}
            for cluster_id in range(n_clusters):
                timesteps_in_cluster = [ts for _, ts in cluster_timestep_distribution[cluster_id]]
                if timesteps_in_cluster:
                    # 计算该聚类中时间步的重要性分数
                    importance_scores = [self.timestep_importance_scores[ts] for ts in timesteps_in_cluster]
                    cluster_importance[cluster_id] = {
                        'mean_importance': np.mean(importance_scores),
                        'std_importance': np.std(importance_scores),
                        'timestep_count': len(timesteps_in_cluster),
                        'unique_timesteps': len(set(timesteps_in_cluster)),
                        'timesteps': timesteps_in_cluster
                    }
            
            hierarchical_results[level] = {
                'n_clusters': n_clusters,
                'cluster_centers': cluster_centers,
                'cluster_labels': cluster_labels,
                'cluster_timestep_distribution': dict(cluster_timestep_distribution),
                'cluster_importance': cluster_importance,
                'kmeans_model': kmeans
            }
            
            print(f"完成 {n_clusters} 聚类分析")
        
        # 找出层次化关键动作
        self.hierarchical_key_actions = self._find_hierarchical_key_actions(hierarchical_results)
        
        return hierarchical_results
    
    def _find_hierarchical_key_actions(self, hierarchical_results):
        """
        从层次化聚类结果中找出关键动作（改进版本）
        
        Args:
            hierarchical_results: 层次化聚类结果
            
        Returns:
            关键动作字典
        """
        key_actions = {}
        
        for level, results in hierarchical_results.items():
            n_clusters = results['n_clusters']
            cluster_importance = results['cluster_importance']
            
            # 根据重要性排序聚类
            sorted_clusters = sorted(
                cluster_importance.items(),
                key=lambda x: x[1]['mean_importance'],
                reverse=True
            )
            
            # 动态选择聚类数量，避免过于严格
            num_clusters_to_select = min(max(2, n_clusters // 2), n_clusters)
            top_clusters = sorted_clusters[:num_clusters_to_select]
            
            level_key_actions = []
            selected_timesteps = set()  # 避免重复选择相同时间步
            
            for cluster_id, importance_info in top_clusters:
                # 获取该聚类中的时间步
                timesteps = importance_info['timesteps']
                unique_timesteps = list(set(timesteps))
                
                # 计算每个唯一时间步的综合分数（频率 + 重要性）
                timestep_scores = {}
                for ts in unique_timesteps:
                    if ts not in selected_timesteps:  # 避免重复
                        frequency = timesteps.count(ts)
                        importance = self.timestep_importance_scores[ts]
                        # 综合分数：归一化频率 + 重要性分数
                        normalized_freq = frequency / len(timesteps)
                        timestep_scores[ts] = 0.6 * importance + 0.4 * normalized_freq
                
                # 选择分数最高的时间步
                if timestep_scores:
                    key_timestep = max(timestep_scores.items(), key=lambda x: x[1])[0]
                    selected_timesteps.add(key_timestep)
                    
                    level_key_actions.append({
                        'cluster_id': cluster_id,
                        'key_timestep': key_timestep,
                        'frequency': timesteps.count(key_timestep),
                        'importance_score': self.timestep_importance_scores[key_timestep],
                        'combined_score': timestep_scores[key_timestep],
                        'cluster_center': results['cluster_centers'][cluster_id]
                    })
            
            # 如果仍然没有足够的多样性，添加基于重要性的补充选择
            if len(level_key_actions) < 2:
                # 从所有时间步中选择重要性最高的几个（排除已选择的）
                all_timesteps = list(range(30))
                remaining_timesteps = [ts for ts in all_timesteps if ts not in selected_timesteps]
                
                # 按重要性排序
                remaining_with_scores = [(ts, self.timestep_importance_scores[ts]) for ts in remaining_timesteps]
                remaining_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 添加前几个最重要的时间步
                for ts, score in remaining_with_scores[:max(1, 3-len(level_key_actions))]:
                    level_key_actions.append({
                        'cluster_id': -1,  # 标记为补充选择
                        'key_timestep': ts,
                        'frequency': 0,
                        'importance_score': score,
                        'combined_score': score,
                        'cluster_center': None
                    })
            
            key_actions[level] = level_key_actions
            
            print(f"\n层级 {level+1} 关键动作:")
            for action in level_key_actions:
                if action['cluster_id'] == -1:
                    print(f"  补充选择: 时间步 {action['key_timestep']} "
                          f"(重要性: {action['importance_score']:.4f})")
                else:
                    print(f"  聚类 {action['cluster_id']}: 时间步 {action['key_timestep']} "
                          f"(频率: {action['frequency']}, 重要性: {action['importance_score']:.4f}, "
                          f"综合分数: {action['combined_score']:.4f})")
        
        return key_actions
    
    def find_consistent_key_actions(self, min_levels=None):
        """
        找出在多个聚类层级中都出现的一致关键动作（改进版本）
        
        Args:
            min_levels: 最少出现的层级数，默认为总层级数的一半
            
        Returns:
            一致关键动作的时间步列表
        """
        if not self.hierarchical_key_actions:
            print("请先运行层次化聚类分析")
            return []
        
        total_levels = len(self.hierarchical_key_actions)
        if min_levels is None:
            min_levels = max(1, total_levels // 2)  # 至少在一半的层级中出现
        
        # 统计每个时间步在各层级中的出现次数
        timestep_counts = {}
        for level, actions in self.hierarchical_key_actions.items():
            level_timesteps = [action['key_timestep'] for action in actions]
            for ts in level_timesteps:
                timestep_counts[ts] = timestep_counts.get(ts, 0) + 1
        
        # 找出出现次数达到要求的时间步
        consistent_timesteps = [ts for ts, count in timestep_counts.items() if count >= min_levels]
        
        # 按重要性排序
        consistent_with_importance = [(ts, self.timestep_importance_scores[ts]) for ts in consistent_timesteps]
        consistent_with_importance.sort(key=lambda x: x[1], reverse=True)
        
        result = [ts for ts, _ in consistent_with_importance]
        
        print(f"\n=== 一致关键动作分析（改进版） ===")
        print(f"在至少 {min_levels}/{total_levels} 个聚类层级中出现的关键时间步: {result}")
        print("\n各时间步出现统计:")
        for ts, count in sorted(timestep_counts.items()):
            print(f"  时间步 {ts}: 出现在 {count}/{total_levels} 个层级中 (重要性: {self.timestep_importance_scores[ts]:.4f})")
        
        return result
    
    def visualize_hierarchical_clustering(self, save_path=None):
        """
        可视化层次化聚类结果
        
        Args:
            save_path: 保存路径
        """
        if not self.hierarchical_key_actions:
            print("请先运行层次化聚类分析")
            return
        
        num_levels = len(self.hierarchical_key_actions)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. 各层级关键时间步分布
        level_names = []
        key_timesteps_by_level = []
        
        for level, actions in self.hierarchical_key_actions.items():
            level_names.append(f"层级{level+1}")
            timesteps = [action['key_timestep'] for action in actions]
            key_timesteps_by_level.extend(timesteps)
        
        axes[0].hist(key_timesteps_by_level, bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_title('层次化关键时间步分布')
        axes[0].set_xlabel('时间步')
        axes[0].set_ylabel('频次')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 各层级关键动作重要性分数
        level_importance_scores = []
        level_labels = []
        
        for level, actions in self.hierarchical_key_actions.items():
            for action in actions:
                level_importance_scores.append(action['importance_score'])
                level_labels.append(f"L{level+1}")
        
        # 按层级分组绘制箱线图
        level_groups = {}
        for i, label in enumerate(level_labels):
            if label not in level_groups:
                level_groups[label] = []
            level_groups[label].append(level_importance_scores[i])
        
        box_data = [level_groups[label] for label in sorted(level_groups.keys())]
        box_labels = sorted(level_groups.keys())
        
        axes[1].boxplot(box_data, labels=box_labels)
        axes[1].set_title('各层级关键动作重要性分数')
        axes[1].set_xlabel('聚类层级')
        axes[1].set_ylabel('重要性分数')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 关键时间步在原始重要性曲线上的标注
        axes[2].plot(range(30), self.timestep_importance_scores, 'b-', linewidth=2, alpha=0.7, label='所有时间步')
        
        # 标注各层级的关键时间步
        colors = ['red', 'green', 'orange', 'purple']
        for level, actions in self.hierarchical_key_actions.items():
            timesteps = [action['key_timestep'] for action in actions]
            scores = [self.timestep_importance_scores[ts] for ts in timesteps]
            color = colors[level % len(colors)]
            axes[2].scatter(timesteps, scores, color=color, s=100, 
                          label=f'层级{level+1}关键点', zorder=5)
        
        axes[2].set_title('关键时间步在重要性曲线上的分布')
        axes[2].set_xlabel('时间步')
        axes[2].set_ylabel('重要性分数')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. 一致关键动作分析
        consistent_timesteps = self.find_consistent_key_actions()
        if consistent_timesteps:
            consistent_scores = [self.timestep_importance_scores[ts] for ts in consistent_timesteps]
            axes[3].bar(range(len(consistent_timesteps)), consistent_scores, 
                       color='gold', alpha=0.8, edgecolor='black')
            axes[3].set_title('一致关键动作重要性分数')
            axes[3].set_xlabel('一致关键时间步索引')
            axes[3].set_ylabel('重要性分数')
            axes[3].set_xticks(range(len(consistent_timesteps)))
            axes[3].set_xticklabels([f'T{ts}' for ts in consistent_timesteps])
        else:
            axes[3].text(0.5, 0.5, '未找到一致关键动作', ha='center', va='center', 
                        transform=axes[3].transAxes, fontsize=14)
            axes[3].set_title('一致关键动作分析')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"层次化聚类可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def get_key_actions_summary(self):
        """
        获取关键动作分析总结
        
        Returns:
            总结字典
        """
        if not self.hierarchical_key_actions:
            return {"error": "请先运行层次化聚类分析"}
        
        summary = {
            "hierarchical_analysis": {},
            "consistent_key_actions": self.find_consistent_key_actions(),
            "total_levels": len(self.hierarchical_key_actions)
        }
        
        for level, actions in self.hierarchical_key_actions.items():
            level_summary = {
                "key_timesteps": [action['key_timestep'] for action in actions],
                "importance_scores": [action['importance_score'] for action in actions],
                "frequencies": [action['frequency'] for action in actions],
                "cluster_ids": [action['cluster_id'] for action in actions]
            }
            summary["hierarchical_analysis"][f"level_{level+1}"] = level_summary
        
        print("\n=== 关键动作分析总结 ===")
        print(f"总聚类层级数: {summary['total_levels']}")
        print(f"一致关键动作: {summary['consistent_key_actions']}")
        
        for level_name, level_data in summary["hierarchical_analysis"].items():
            print(f"\n{level_name}:")
            print(f"  关键时间步: {level_data['key_timesteps']}")
            print(f"  重要性分数: {[f'{score:.4f}' for score in level_data['importance_scores']]}")
        
        return summary

# 使用示例
def main():
    # 初始化分析器
    h5_file_path = "result/robot_trajectory_data2.h5"
    analyzer = TimestepImportanceAnalyzer(h5_file_path)
    
    # 加载数据
    analyzer.load_data()
    
    # 计算时间步重要性
    analyzer.calculate_timestep_importance()
    
    # 找出重要时间步
    print("\n=== 使用top_k方法找出6个最重要的时间步 ===")
    important_timesteps = analyzer.find_important_timesteps(num_important=6, method='top_k')
    
    # 层次化聚类分析
    print("\n=== 开始层次化聚类分析 ===")
    hierarchical_results = analyzer.hierarchical_clustering_analysis(
        cluster_centers_list=[20, 10, 5, 3]
    )
    
    # 获取关键动作总结
    summary = analyzer.get_key_actions_summary()
    
    # 可视化层次化聚类结果
    analyzer.visualize_hierarchical_clustering(save_path="hierarchical_clustering_analysis.png")
    
    # 原有的可视化
    analyzer.visualize_timestep_importance(save_path="timestep_importance_analysis.png")
    
    return analyzer, summary

if __name__ == "__main__":
    analyzer, summary = main()
