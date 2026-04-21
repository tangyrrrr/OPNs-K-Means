import math
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
import time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn import datasets as sk_datasets
from sklearn.preprocessing import LabelEncoder
from data_loader.dataset_loader import DatasetLoader

# 引入多进程所需模块
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 将数据集映射提取为全局变量，以便多进程的 worker 能够访问
datasets_map = {
    'iris': DatasetLoader.iris, 'balance-scale': DatasetLoader.balance, 'weather': DatasetLoader.weather,
    'hayes_roth': DatasetLoader.hayes_roth, 'phoneme': DatasetLoader.phoneme, 'monk-2': DatasetLoader.monk_2,
    'led7digit': DatasetLoader.led7digit, 'appendicitis': DatasetLoader.appendicitis, 'ecoli': DatasetLoader.ecoli,
    'pima': DatasetLoader.pima, 'cars': DatasetLoader.cars, 'saheart': DatasetLoader.saheart,
    'heart': DatasetLoader.heart, 'cleve': DatasetLoader.cleve, 'cleveland': DatasetLoader.cleveland,
    'wine': DatasetLoader.wine, 'vowel': DatasetLoader.vowel, 'penbased': DatasetLoader.penbased,
    'vehicle': DatasetLoader.vehicle, 'hepatitis': DatasetLoader.hepatitis, 'segment': DatasetLoader.segment,
    'sonar': DatasetLoader.sonar, 'air': DatasetLoader.air
}

def m_distance(x, y):
    return np.sum(np.abs(x - y))

def kmeans_plusplus_m(X, k):
    n_samples = X.shape[0]
    centroids = [X[random.randint(0, n_samples - 1)]]
    for _ in range(1, k):
        D = np.array([min([m_distance(x, c) for c in centroids]) for x in X])
        probs = D / np.sum(D) if np.sum(D) > 0 else np.ones(n_samples) / n_samples
        cumulative_probs = np.cumsum(probs)
        centroids.append(X[min(np.searchsorted(cumulative_probs, random.random()), n_samples - 1)])
    return np.array(centroids)

# 增强版：曼哈顿距离 K-means
def kmeans_M(X, k, max_iter=100, n_init=5):
    n_samples, n_features = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_clusters, best_centroids, best_y_pred = None, None, None
    best_total_dist = float('inf')

    for _ in range(n_init):
        centroids = kmeans_plusplus_m(X_scaled, k)
        for _ in range(max_iter):
            clusters = [[] for _ in range(k)]
            total_dist = 0.0
            
            for idx in range(n_samples):
                sample = X_scaled[idx]
                distances = [m_distance(sample, centroid) for centroid in centroids]
                cluster_idx = np.argmin(distances) # 曼哈顿距离取最小
                clusters[cluster_idx].append(idx)
                total_dist += distances[cluster_idx]

            new_centroids_list = []
            for cluster in clusters:
                if len(cluster) > 0:
                    new_centroids_list.append(np.mean(X_scaled[cluster], axis=0))
                else:
                    new_centroids_list.append(X_scaled[random.choice(range(n_samples))])
            new_centroids = np.array(new_centroids_list)

            if np.allclose(centroids, new_centroids, atol=1e-4): break
            centroids = new_centroids

        # 曼哈顿距离越小越好
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_clusters, best_centroids = clusters, centroids
            best_y_pred = np.zeros(n_samples, dtype=int)
            for c_idx, indices in enumerate(best_clusters):
                for idx in indices: best_y_pred[idx] = c_idx

    return best_clusters, best_centroids, best_y_pred

def cluster_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    unique_true, unique_pred = np.unique(y_true), np.unique(y_pred)
    cost_matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=np.int64)
    for i in range(len(unique_true)):
        for j in range(len(unique_pred)):
            cost_matrix[i, j] = np.sum(y_pred[y_true == unique_true[i]] == unique_pred[j])
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / y_true.size if y_true.size > 0 else 0.0

# 单次实验任务（供多进程调用）
def run_single_experiment(run_idx, dataset_key):
    loader = datasets_map[dataset_key]
    try:
        X, y, data_name, n_features, k_classes = loader()
        if X is None: return None
        
        y_encoded = LabelEncoder().fit_transform(y)
        
        start_time = time.time()
        clusters, centroids, y_pred = kmeans_M(X, k_classes)
        elapsed = time.time() - start_time

        acc = cluster_accuracy(y_encoded, y_pred)
        ari = adjusted_rand_score(y_encoded, y_pred)
        recall = recall_score(y_encoded, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_encoded, y_pred, average='macro', zero_division=0)

        return {
            '运行次数': run_idx, 
            '数据集键名': dataset_key, # 内部使用，处理断点逻辑
            '数据集': data_name, 
            'ARI': ari, 
            '准确率': acc, 
            '召回率': recall, 
            'F1值': f1, 
            '收敛时间(s)': elapsed, 
            '耗时(s)': elapsed
        }
    except Exception as e:
        logger.error(f"Error on {dataset_key} Run {run_idx}: {str(e)}")
        return {'error': True}

if __name__ == "__main__":
    num_runs = 10
    output_dir = "result_M"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    results_csv_path = os.path.join(output_dir, "all_runs_details.csv")
    summary_csv_path = os.path.join(output_dir, "result_summary_mean_std.csv")

    all_results = pd.read_csv(results_csv_path).to_dict('records') if os.path.exists(results_csv_path) else []

    # 1. 收集所有未完成的任务
    tasks_to_run = []
    for run in range(1, num_runs + 1):
        for name in datasets_map.keys():
            # 兼容处理：检查任务是否已存在于 CSV 中
            if any(r['运行次数'] == run and (r.get('数据集键名', r.get('数据集')) == name) for r in all_results):
                continue
            tasks_to_run.append((run, name))

    if not tasks_to_run:
        logger.info("所有数据集的运行任务已全部完成！即将生成最终汇总。")
    else:
        # 2. 拉满 CPU 并行运行任务
        max_workers = multiprocessing.cpu_count()
        logger.info(f"发现 {len(tasks_to_run)} 个任务待运行。正在启动多进程池，核心数拉满: {max_workers} ...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(run_single_experiment, run, name): (run, name) for run, name in tasks_to_run}
            
            # 使用 as_completed 边跑边收集，主进程安全写入 CSV
            for future in as_completed(future_to_task):
                result = future.result()
                if result and not result.get('error'):
                    # 移除为了断点续跑设置的辅助键，避免污染原本的 CSV 格式
                    result_to_save = result.copy()
                    result_to_save.pop('数据集键名', None)
                    
                    all_results.append(result_to_save)
                    logger.info(f"[{result['数据集']} Run {result['运行次数']}] ARI: {result['ARI']:.4f}, ACC: {result['准确率']:.4f}")
                    
                    # 实时增量保存，断点续跑机制依旧生效
                    pd.DataFrame(all_results).to_csv(results_csv_path, index=False, encoding='utf-8-sig')

    # 3. 汇总数据
    if all_results:
        df = pd.DataFrame(all_results)
        summary_list = []
        for name, g in df.groupby('数据集'):
            if not g.empty:
                summary_row = {'数据集': name}
                for col in ['ARI', '准确率', '召回率', 'F1值', '收敛时间(s)', '耗时(s)']:
                    summary_row[col] = f"{g[col].mean():.4f} ± {g[col].std():.4f}"
                summary_list.append(summary_row)
        
        pd.DataFrame(summary_list).to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"已生成并覆盖最终数据统计: {summary_csv_path}")