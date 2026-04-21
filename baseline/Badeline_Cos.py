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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 余弦相似度计算
def cos_similarity(x, y):
    num = np.dot(x, y)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den == 0: return 0.0
    return num / den

# 余弦距离 K-means++ 初始化 (距离 = 1 - 相似度)
def kmeans_plusplus_cos(X, k):
    n_samples = X.shape[0]
    centroids = [X[random.randint(0, n_samples - 1)]]
    for _ in range(1, k):
        D2 = np.array([min([max(1.0 - cos_similarity(x, c), 0.0) for c in centroids]) for x in X])
        probs = D2 / np.sum(D2) if np.sum(D2) > 0 else np.ones(n_samples) / n_samples
        cumulative_probs = np.cumsum(probs)
        centroids.append(X[min(np.searchsorted(cumulative_probs, random.random()), n_samples - 1)])
    return np.array(centroids)

# 增强版：余弦 K-means
def kmeans_Cos(X, k, max_iter=100, n_init=5):
    n_samples, n_features = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_clusters, best_centroids, best_y_pred = None, None, None
    best_total_sim = -float('inf')

    for _ in range(n_init):
        centroids = kmeans_plusplus_cos(X_scaled, k)
        for _ in range(max_iter):
            clusters = [[] for _ in range(k)]
            total_sim = 0.0
            
            for idx in range(n_samples):
                sample = X_scaled[idx]
                similarities = [cos_similarity(sample, centroid) for centroid in centroids]
                cluster_idx = np.argmax(similarities) # 相似度最大
                clusters[cluster_idx].append(idx)
                total_sim += similarities[cluster_idx]

            new_centroids_list = []
            for cluster in clusters:
                if len(cluster) > 0:
                    new_centroids_list.append(np.mean(X_scaled[cluster], axis=0))
                else:
                    new_centroids_list.append(X_scaled[random.choice(range(n_samples))])
            new_centroids = np.array(new_centroids_list)

            if np.allclose(centroids, new_centroids, atol=1e-4): break
            centroids = new_centroids

        if total_sim > best_total_sim:
            best_total_sim = total_sim
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

if __name__ == "__main__":
    datasets = {
        'iris': DatasetLoader.iris, 'balance-scale': DatasetLoader.balance, 'weather': DatasetLoader.weather,
        'hayes_roth': DatasetLoader.hayes_roth, 'phoneme': DatasetLoader.phoneme, 'monk-2': DatasetLoader.monk_2,
        'led7digit': DatasetLoader.led7digit, 'appendicitis': DatasetLoader.appendicitis, 'ecoli': DatasetLoader.ecoli,
        'pima': DatasetLoader.pima, 'cars': DatasetLoader.cars, 'saheart': DatasetLoader.saheart,
        'heart': DatasetLoader.heart, 'cleve': DatasetLoader.cleve, 'cleveland': DatasetLoader.cleveland,
        'wine': DatasetLoader.wine, 'vowel': DatasetLoader.vowel, 'penbased': DatasetLoader.penbased,
        'vehicle': DatasetLoader.vehicle, 'hepatitis': DatasetLoader.hepatitis, 'segment': DatasetLoader.segment,
        'sonar': DatasetLoader.sonar, 'air': DatasetLoader.air
    }

    num_runs = 10
    output_dir = "result_Cos"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    results_csv_path = os.path.join(output_dir, "all_runs_details.csv")
    summary_csv_path = os.path.join(output_dir, "result_summary_mean_std.csv")

    all_results = pd.read_csv(results_csv_path).to_dict('records') if os.path.exists(results_csv_path) else []

    for run in range(1, num_runs + 1):
        for name, loader in datasets.items():
            try:
                if any(r['运行次数'] == run and r['数据集'] == name for r in all_results): continue
                X, y, data_name, n_features, k_classes = loader()
                if X is None: continue
                y_encoded = LabelEncoder().fit_transform(y)
                
                start_time = time.time()
                clusters, centroids, y_pred = kmeans_Cos(X, k_classes)
                elapsed = time.time() - start_time

                acc = cluster_accuracy(y_encoded, y_pred)
                ari = adjusted_rand_score(y_encoded, y_pred)
                recall = recall_score(y_encoded, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_encoded, y_pred, average='macro', zero_division=0)

                all_results.append({'运行次数': run, '数据集': data_name, 'ARI': ari, '准确率': acc, '召回率': recall, 'F1值': f1, '收敛时间(s)': elapsed, '耗时(s)': elapsed})
                logger.info(f"[{data_name} Run {run}] ARI: {ari:.4f}, ACC: {acc:.4f}")
                pd.DataFrame(all_results).to_csv(results_csv_path, index=False, encoding='utf-8-sig')
            except Exception as e:
                logger.error(f"Error on {name}: {str(e)}")

    if all_results:
        df = pd.DataFrame(all_results)
        summary_list = [{'数据集': name, **{col: f"{g[col].mean():.4f} ± {g[col].std():.4f}" for col in ['ARI', '准确率', '召回率', 'F1值', '收敛时间(s)', '耗时(s)']}} for name, g in df.groupby('数据集') if not g.empty]
        pd.DataFrame(summary_list).to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        logger.info("测试完成并已汇总。")