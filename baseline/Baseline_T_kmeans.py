import math
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import time
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets as sk_datasets
from sklearn.preprocessing import LabelEncoder
import scipy.io

from data_loader.dataset_loader import DatasetLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 谷本系数距离计算
def T_distance(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    numerator = np.sum(x * y)
    denominator = np.sum(x ** 2) + np.sum(y ** 2) - numerator
    # 处理分母为零的情况
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


# K-means++ 初始化策略
def kmeans_plusplus_initialization(X, k):
    n_samples = X.shape[0]
    # 1. 随机选择第一个中心点
    centroids = [X[random.randint(0, n_samples - 1)]]
    
    for _ in range(1, k):
        # 计算每个样本到最近已有中心点的距离（使用 1 - 相似度 作为距离）
        D2 = np.array([min([1.0 - T_distance(x, c) for c in centroids]) for x in X])
        D2 = np.maximum(D2, 0)  # 确保数值稳定，不出现负数
        
        # 计算被选为下一个中心点的概率
        if np.sum(D2) > 0:
            probs = D2 / np.sum(D2)
        else:
            probs = np.ones(n_samples) / n_samples
            
        # 轮盘赌算法按概率选取
        cumulative_probs = np.cumsum(probs)
        r = random.random()
        next_centroid_idx = np.searchsorted(cumulative_probs, r)
        next_centroid_idx = min(next_centroid_idx, n_samples - 1) # 防止极小概率下的索引越界
        centroids.append(X[next_centroid_idx])
        
    return np.array(centroids)


# 修改后：谷本系数 K-means 算法 (增加 n_init 并使用 MinMaxScaler)
def kmeans_T(X, k, max_iter=100, n_init=5):
    n_samples, n_features = X.shape

    # 数据标准化：使用 MinMaxScaler 确保特征非负，适配谷本系数
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    best_clusters = None
    best_centroids = None
    best_y_pred = None
    best_total_sim = -float('inf')

    # 多次初始化机制，取内部质量最好的一次
    for init_run in range(n_init):
        # 使用 K-means++ 初始化
        centroids = kmeans_plusplus_initialization(X_scaled, k)
        
        for _ in range(max_iter):
            clusters = [[] for _ in range(k)]
            total_sim = 0.0
            
            # 分配样本到最近的中心点（谷本系数最大，因为它是相似度）
            for idx in range(n_samples):
                sample = X_scaled[idx]
                similarities = [T_distance(sample, centroid) for centroid in centroids]
                cluster_idx = np.argmax(similarities)  # 相似度最大
                clusters[cluster_idx].append(idx)
                total_sim += similarities[cluster_idx]

            # 更新中心点
            new_centroids_list = []
            for cluster in clusters:
                if len(cluster) > 0:
                    new_centroid = np.mean(X_scaled[cluster], axis=0)
                else:
                    # 处理空簇
                    new_centroid = X_scaled[random.choice(range(n_samples))]
                new_centroids_list.append(new_centroid)

            new_centroids = np.array(new_centroids_list)

            # 检查收敛
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        # 评估本次初始化的聚类质量（簇内总相似度越大越好）
        if total_sim > best_total_sim:
            best_total_sim = total_sim
            best_clusters = clusters
            best_centroids = centroids
            
            best_y_pred = np.zeros(n_samples, dtype=int)
            for cluster_idx, indices in enumerate(best_clusters):
                for sample_idx in indices:
                    best_y_pred[sample_idx] = cluster_idx

    return best_clusters, best_centroids, best_y_pred


# 聚类准确率计算（使用匈牙利算法）
def cluster_accuracy(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    assert y_true.size == y_pred.size

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    n_classes = len(unique_true)
    n_clusters = len(unique_pred)

    cost_matrix = np.zeros((n_classes, n_clusters), dtype=np.int64)
    for i in range(n_classes):
        for j in range(n_clusters):
            mask = (y_true == unique_true[i])
            cost_matrix[i, j] = np.sum(y_pred[mask] == unique_pred[j])

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / y_true.size if y_true.size > 0 else 0.0


# 主程序
if __name__ == "__main__":
    datasets = {
        # 'iris_5an_nn':DatasetLoader.iris_5an_nn,
        # 'mammographic':DatasetLoader.mammographic,
        # 'newthyroid':DatasetLoader.newthyroid,
        # 'yeast':DatasetLoader.yeast,
        # 'diabetes': DatasetLoader.diabetes,
        # 'glass': DatasetLoader.glass,
        # 'WBC': DatasetLoader.WBC,
        # 'page_blocks': DatasetLoader.page_blocks,
        # 'winequality': DatasetLoader.winequality,
        # 'marketing': DatasetLoader.marketing,
        # 'austra': DatasetLoader.austra,
        # 'vote': DatasetLoader.vote,
        # 'bands': DatasetLoader.bands,
        'iris': DatasetLoader.iris,
        'balance-scale': DatasetLoader.balance,
        'weather': DatasetLoader.weather,
        'hayes_roth': DatasetLoader.hayes_roth,
        'phoneme': DatasetLoader.phoneme,
        'monk-2': DatasetLoader.monk_2,
        'led7digit': DatasetLoader.led7digit,
        'appendicitis': DatasetLoader.appendicitis,
        'ecoli': DatasetLoader.ecoli,
        'pima': DatasetLoader.pima,
        'cars': DatasetLoader.cars,
        'saheart': DatasetLoader.saheart,
        'heart': DatasetLoader.heart,
        'cleve': DatasetLoader.cleve,
        'cleveland': DatasetLoader.cleveland,
        'wine': DatasetLoader.wine,
        'vowel': DatasetLoader.vowel,
        'penbased': DatasetLoader.penbased,
        'vehicle': DatasetLoader.vehicle,
        'hepatitis': DatasetLoader.hepatitis,
        'segment': DatasetLoader.segment,
        'sonar': DatasetLoader.sonar,
        'air': DatasetLoader.air
    }

    num_runs = 10  # 每个数据集运行次数
    # 1. 定义结果文件夹名称
    output_dir = "result_T"
    
    # 2. 检查并创建文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"已创建结果文件夹: '{output_dir}'")

    # 3. 更新结果文件的保存路径
    results_csv_path = os.path.join(output_dir, "all_runs_details.csv")
    summary_csv_path = os.path.join(output_dir, "result_summary_mean_std.csv")

    # --- 断点续跑功能 ---
    if os.path.exists(results_csv_path):
        logger.info(f"发现已存在的结果文件 '{results_csv_path}'，将加载并继续。")
        all_results_df = pd.read_csv(results_csv_path)
        all_results = all_results_df.to_dict('records')
    else:
        logger.info(f"未发现结果文件 '{results_csv_path}'，将从头开始运行。")
        all_results = []
    # --- 断点续跑功能结束 ---

    for run in range(1, num_runs + 1):
        logger.info(f"\n=== 开始第 {run}/{num_runs} 次运行 ===")
        for name, loader in datasets.items():
            try:
                # 检查此运行/数据集组合是否已完成
                is_completed = any(
                    r['运行次数'] == run and r['数据集'] == name for r in all_results
                )
                if is_completed:
                    logger.info(f"跳过已完成的任务: [运行 {run}, 数据集 {name}]")
                    continue

                X, y, data_name, n_features, k_classes = loader()
                if X is None: continue

                y_encoded = LabelEncoder().fit_transform(y)

                logger.info(f"处理 {data_name} (特征数={n_features}, 类别数={k_classes})")

                start_time = time.time()
                clusters, centroids, y_pred = kmeans_T(X, k_classes)
                elapsed = time.time() - start_time

                acc = cluster_accuracy(y_encoded, y_pred)
                ari = adjusted_rand_score(y_encoded, y_pred)
                recall = recall_score(y_encoded, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_encoded, y_pred, average='macro', zero_division=0)

                result = {
                    '运行次数': run, '数据集': data_name, 'ARI': ari,
                    '准确率': acc, '召回率': recall, 'F1值': f1, 
                    '收敛时间(s)': elapsed, '耗时(s)': elapsed,
                }
                all_results.append(result)
                logger.info(f"结果 - ARI: {ari:.4f}, 准确率: {acc:.4f}, 收敛时间: {elapsed:.2f}s")

                pd.DataFrame(all_results).to_csv(results_csv_path, index=False, encoding='utf-8-sig')

            except Exception as e:
                logger.error(f"处理数据集 {name} 时失败: {str(e)}", exc_info=True)

    # --- 所有运行结束后，进行最终的统计和总结 ---
    if all_results:
        df = pd.DataFrame(all_results)

        summary_list = []
        for dataset_name, group in df.groupby('数据集'):
            if group.empty: continue

            avg = group.mean(numeric_only=True)
            std = group.std(numeric_only=True)

            summary_row = {'数据集': dataset_name}
            for col in ['ARI', '准确率', '召回率', 'F1值']:
                summary_row[col] = f"{avg[col]:.4f} ± {std[col]:.4f}"
            
            # 添加时间指标
            summary_row['收敛时间(s)'] = f"{avg['收敛时间(s)']:.2f} ± {std['收敛时间(s)']:.2f}"
            summary_row['耗时(s)'] = f"{avg['耗时(s)']:.2f} ± {std['耗时(s)']:.2f}"

            summary_list.append(summary_row)

        final_summary_df = pd.DataFrame(summary_list).reindex(
            columns=['数据集', 'ARI', '准确率', '召回率', 'F1值', '收敛时间(s)', '耗时(s)']
        )

        final_summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')

        logger.info("\n" + "=" * 20 + " 最终结果汇总 (平均值 ± 标准差) " + "=" * 20)
        print(final_summary_df.to_string(index=False))
    else:
        logger.warning("所有任务均未产生有效结果。")