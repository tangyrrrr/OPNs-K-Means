import pandas as pd
import numpy as np
import time
import os
import logging
import functools
from tqdm import tqdm
from scipy.stats import mode
from multiprocessing import Pool, cpu_count

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (adjusted_rand_score, accuracy_score, 
                             recall_score, f1_score, normalized_mutual_info_score)
from sklearn.metrics.pairwise import pairwise_kernels

# 从您的项目中导入相同的 DataLoader
from data_loader import DatasetLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 核心算法：Kernel K-Means (二阶多项式核) - 支持 K-means++ 和 多次重启
# ==============================================================================
class KernelKMeans:
    """
    标准的 Kernel K-means 实现。
    直接在核矩阵 K 上进行距离计算和聚类分配。
    """
    # 【修改点 1】：新增 n_init 和 init 参数，默认为 n_init=5 和 init='k-means++'
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None, init='k-means++', n_init=5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.n_iter_ = 0  # 记录实际收敛所需的迭代次数

    def fit(self, K):
        n_samples = K.shape[0]
        rng = np.random.RandomState(self.random_state)
        
        K_diag = np.diag(K) # 预计算对角线元素 K_{ii}
        
        best_inertia = np.inf
        best_labels = None
        best_n_iter = 0
        
        # 【修改点 2】：引入 n_init 循环，运行多次取最优
        for init_run in range(self.n_init):
            
            # 【修改点 3】：基于核空间的 K-means++ 初始化
            if self.init == 'k-means++':
                centers = [rng.randint(n_samples)] # 随机挑第一个中心
                min_dist_sq = np.full(n_samples, np.inf)
                
                for _ in range(1, self.n_clusters):
                    last_c = centers[-1]
                    # 核空间距离公式: d^2(x_i, c) = K_ii + K_cc - 2K_ic
                    dist_sq = K_diag + K_diag[last_c] - 2 * K[:, last_c]
                    dist_sq = np.maximum(dist_sq, 0.0) # 防止浮点数精度引发负数
                    
                    min_dist_sq = np.minimum(min_dist_sq, dist_sq)
                    total_dist = np.sum(min_dist_sq)
                    
                    if total_dist > 0:
                        probs = min_dist_sq / total_dist
                        next_c = rng.choice(n_samples, p=probs)
                    else:
                        avail = list(set(range(n_samples)) - set(centers))
                        next_c = rng.choice(avail) if avail else rng.randint(n_samples)
                        
                    centers.append(next_c)
                    
                # 将每个样本分配到距离最近的初始中心
                dist_init = np.zeros((n_samples, self.n_clusters))
                for c_idx, c in enumerate(centers):
                    dist_init[:, c_idx] = K_diag + K_diag[c] - 2 * K[:, c]
                labels = np.argmin(dist_init, axis=1)
                
            else:
                # 备用：纯随机初始化类分配
                labels = rng.randint(self.n_clusters, size=n_samples)

            dist = np.zeros((n_samples, self.n_clusters))
            n_iter_current = 0
            
            for it in range(self.max_iter):
                dist.fill(0)
                
                # 距离计算: ||phi(x) - mu_c||^2 = K_ii - 2/Nc * sum(K_ic) + 1/Nc^2 * sum(K_cc)
                for c in range(self.n_clusters):
                    mask = (labels == c)
                    N_c = np.sum(mask)
                    
                    # 处理空簇
                    if N_c == 0:
                        dist[:, c] = np.inf
                        continue
                    
                    term3 = np.sum(K[mask][:, mask]) / (N_c ** 2)
                    term2 = 2 * np.sum(K[:, mask], axis=1) / N_c
                    
                    # K_ii 恒定，省略不影响 argmin
                    dist[:, c] = -term2 + term3
                
                new_labels = np.argmin(dist, axis=1)
                
                # 处理空簇
                empty_clusters = np.where(np.bincount(new_labels, minlength=self.n_clusters) == 0)[0]
                if len(empty_clusters) > 0:
                    largest_cluster = np.argmax(np.bincount(new_labels, minlength=self.n_clusters))
                    candidates = np.where(new_labels == largest_cluster)[0]
                    for ec in empty_clusters:
                        if len(candidates) > 1:
                            idx = rng.choice(candidates)
                            new_labels[idx] = ec
                            candidates = np.where(new_labels == largest_cluster)[0]
                            
                # 判断是否收敛
                if np.array_equal(labels, new_labels):
                    n_iter_current = it + 1
                    break
                    
                labels = new_labels
                n_iter_current = it + 1
                
            # 【修改点 4】：计算当前次初始化的相对惯性 (Inertia)
            # 完整距离平方 = K_ii + (-term2 + term3)，加上常量 K_ii 可得到实际惯性
            inertia = np.sum(K_diag)
            for c in range(self.n_clusters):
                mask = (labels == c)
                if np.sum(mask) > 0:
                    inertia += np.sum(dist[mask, c])
                    
            # 记录 n_init 次中最好的一次
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_n_iter = n_iter_current
                
        # 最终赋值
        self.labels_ = best_labels
        self.n_iter_ = best_n_iter
        
        return self

# ==============================================================================
# 评估指标与多进程顶层辅助函数
# ==============================================================================
def evaluate_clustering(y_true, y_pred):
    unique_clusters = np.unique(y_pred)
    label_mapping = {}

    for cluster_id in unique_clusters:
        mask = (y_pred == cluster_id)
        if np.sum(mask) == 0:
            continue
        true_labels = y_true[mask]
        if len(true_labels) > 0:
            mode_res = mode(true_labels, keepdims=False)
            majority_label = mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0]
            if isinstance(majority_label, np.ndarray):
                majority_label = majority_label[0]
            label_mapping[cluster_id] = majority_label

    aligned_labels = np.zeros_like(y_pred)
    for cluster_id, true_label in label_mapping.items():
        aligned_labels[y_pred == cluster_id] = true_label

    accuracy = accuracy_score(y_true, aligned_labels)
    recall = recall_score(y_true, aligned_labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, aligned_labels, average='macro', zero_division=0)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred) 

    return {'ARI': ari, 'NMI': nmi, 'Accuracy': accuracy, 'Recall': recall, 'F1': f1}

def _run_single_kkm(run_id, K_matrix, k_classes, y_true):
    """顶层函数：专为多进程并发调用设计"""
    start = time.time()
    
    # 【修改点 5】：明确传递 n_init=5, max_iter=100 和 init='k-means++'
    kkm = KernelKMeans(n_clusters=k_classes, max_iter=100, n_init=5, init='k-means++', random_state=(42 + run_id))
    kkm.fit(K_matrix)
    
    elapsed = time.time() - start
    
    scores = evaluate_clustering(y_true, kkm.labels_)
    return {
        '运行次数': run_id,
        '迭代次数': kkm.n_iter_,  
        'ARI': scores['ARI'],
        'NMI': scores['NMI'], 
        'Accuracy': scores['Accuracy'],
        'Recall': scores['Recall'],
        'F1': scores['F1'],
        '耗时(s)': elapsed
    }

# ==============================================================================
# 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    datasets = {
        'iris':DatasetLoader.iris,
        'balance-scale':DatasetLoader.balance,
        'weather':DatasetLoader.weather,
        'hayes_roth':DatasetLoader.hayes_roth,
        'phoneme':DatasetLoader.phoneme,
        'monk-2':DatasetLoader.monk_2,
        'led7digit':DatasetLoader.led7digit,
        'appendicitis':DatasetLoader.appendicitis,
        'ecoli':DatasetLoader.ecoli,
        'pima':DatasetLoader.pima,
        'cars':DatasetLoader.cars,
        'saheart':DatasetLoader.saheart,
        'heart':DatasetLoader.heart,
        'cleve': DatasetLoader.cleve,
        'cleveland':DatasetLoader.cleveland,
        'wine':DatasetLoader.wine,
        'vowel':DatasetLoader.vowel,
        'penbased':DatasetLoader.penbased,
        'vehicle': DatasetLoader.vehicle,
        'hepatitis':DatasetLoader.hepatitis,
        'segment':DatasetLoader.segment,
        'sonar':DatasetLoader.sonar,
        'air':DatasetLoader.air
    }

    num_runs = 10
    n_jobs = min(cpu_count(), num_runs) 
    
    all_detailed_results = []
    all_summary_results = []
    
    detailed_output_file = "second_stability_order_kernel_kmeans_detailed_runs1.csv"
    summary_output_file = "second_stability_order_kernel_kmeans_summary1.csv"

    logger.info(f"🚀 开始运行多进程 Kernel K-means 测试 (二阶核) | 启用核心数: {n_jobs}")

    for name, loader in datasets.items():
        try:
            result_tuple = loader()
            if result_tuple[0] is None: 
                continue
                
            X, y, data_name, n_features, k_classes = result_tuple

            y = LabelEncoder().fit_transform(y)
            n_samples = X.shape[0]

            logger.info(f"\n--- 处理 {data_name} (样本数={n_samples}, 特征数={n_features}, 类别数={k_classes}) ---")

            # 提前标准化并预计算核矩阵，所有进程共享，避免重复计算
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            K_matrix = pairwise_kernels(X_scaled, metric="poly", degree=2, coef0=1.0)

            dataset_results = []
            
            with Pool(processes=n_jobs) as pool:
                eval_func = functools.partial(
                    _run_single_kkm, 
                    K_matrix=K_matrix, 
                    k_classes=k_classes, 
                    y_true=y
                )
                
                pbar = tqdm(pool.imap_unordered(eval_func, range(1, num_runs + 1)), 
                            total=num_runs, 
                            desc=f"并行运行 {data_name} ({num_runs}次)")
                
                for res in pbar:
                    res['数据集'] = data_name
                    res['样本数量'] = n_samples
                    res['特征数量'] = n_features
                    res['类别数量'] = k_classes
                    res['模型'] = 'Kernel_KMeans_Degree2'
                    
                    dataset_results.append(res)
                    pbar.set_postfix_str(f"单次 ARI: {res['ARI']:.4f}")

            # 排序整理确保输出顺序为 1 到 50
            dataset_results = sorted(dataset_results, key=lambda x: x['运行次数'])

            # 将纯数值的详细运行记录保存到详细列表
            all_detailed_results.extend(dataset_results)

            # 汇总该数据集的 50 次统计数据 (均值 ± 标准差)
            if dataset_results:
                aris = [res['ARI'] for res in dataset_results]
                nmis = [res['NMI'] for res in dataset_results]
                accs = [res['Accuracy'] for res in dataset_results]
                recs = [res['Recall'] for res in dataset_results]
                f1s  = [res['F1'] for res in dataset_results]
                iters = [res['迭代次数'] for res in dataset_results]
                times = [res['耗时(s)'] for res in dataset_results]

                stats_result = {
                    '数据集': data_name,
                    '样本数量': n_samples,
                    '特征数量': n_features,
                    '类别数量': k_classes,
                    '迭代次数': f"{np.mean(iters):.4f} ± {np.std(iters):.4f}", 
                    'ARI': f"{np.mean(aris):.4f} ± {np.std(aris):.4f}",
                    'NMI': f"{np.mean(nmis):.4f} ± {np.std(nmis):.4f}",
                    'Accuracy': f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
                    'Recall': f"{np.mean(recs):.4f} ± {np.std(recs):.4f}",
                    'F1': f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
                    '平均耗时(s)': f"{np.mean(times):.4f}", 
                    '总耗时(s)': f"{sum(times):.4f}",
                    '模型': 'Kernel_KMeans_Degree2'
                }

                all_summary_results.append(stats_result)
                logger.info(f"✅ {data_name} 完成. 迭代次数: {stats_result['迭代次数']} | 综合 ARI: {stats_result['ARI']}")

        except Exception as e:
            logger.error(f"❌ {name} 处理失败: {e}", exc_info=True)

    # ==========================================
    # 分别保存为两个 CSV 文件
    # ==========================================
    columns_detailed = ['模型', '数据集', '运行次数', '样本数量', '特征数量', '类别数量', 
                        '迭代次数', 'ARI', 'NMI', 'Accuracy', 'Recall', 'F1', '耗时(s)']
    
    columns_summary = ['模型', '数据集', '样本数量', '特征数量', '类别数量', 
                       '迭代次数', 'ARI', 'NMI', 'Accuracy', 'Recall', 'F1', '平均耗时(s)', '总耗时(s)']

    if all_detailed_results:
        df_detailed = pd.DataFrame(all_detailed_results)[columns_detailed]
        df_detailed.to_csv(detailed_output_file, mode='w', header=True, index=False, encoding='utf-8-sig')
        print(f"\n📄 每次运行的所有详细指标 (纯数值) 已保存至：{detailed_output_file}")

    if all_summary_results:
        df_summary = pd.DataFrame(all_summary_results)[columns_summary]
        df_summary.to_csv(summary_output_file, mode='w', header=True, index=False, encoding='utf-8-sig')
        print(f"📊 最终的统计结果 (均值±标准差) 已保存至：{summary_output_file}")
