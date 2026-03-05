import numpy as np
import os
import random
import time
import pandas as pd
from sklearn import datasets
from .common.gen_pairs import seq_all_pairs_list1, seq_all_pairs_list2, seq_all_pairs_list, seq_all_pairs_with_repeats
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import datasets as sk_datasets

from .data_loader import DatasetLoader
from .opns_pairer import OPNsPairer
from .common import opnpy
from .common.opn import OPNs
import logging
from .common.opn1 import max as opn_max
from tqdm import tqdm
import scipy.io
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from multiprocessing import Pool, cpu_count
import functools
import gc
from multiprocessing import set_start_method
from .common import opnpy

def run_opns_kmeans_a(X, y, k_classes, data_name):
    """
    Performs a single run of the OPNs-K-means(a) algorithm.
    This algorithm is suitable for low-dimensional data and employs a global search strategy to find the optimal feature pair.

    Args:
    X (np.array): The input feature data.
    y (np.array): The true labels.
    k_classes (int): The number of classes.
    data_name (str): The name of the dataset, used for logging.

    Returns:
    dict: A dictionary containing the evaluation metrics and the best feature pair; returns None if it fails.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running OPNs-K-means(a) on {data_name}...")

    # 1. Prepare data
    # Name y for display in the progress bar and encode it
    y_series = pd.Series(y, name=data_name)  
    y_encoded = LabelEncoder().fit_transform(y_series)

    # 2. Core algorithm call
    start_time = time.time()

    # Call the parallel processing function you have already written
    # Note: The n_jobs parameter can be set as needed
    clusters, centers, best_pair, metrics = kmeans_opns_parallel(
        X, 
        k_classes, 
        y_true=y_encoded,
        n_jobs=cpu_count() - 1 # Use multi-core processing
    )
    
    elapsed_time = time.time() - start_time

    # 3. Process and return results
    if metrics and 'ari' in metrics:
        result = {
            'ARI': metrics['ari'],
            'Accuracy': metrics['accuracy'],
            'F1': metrics['f1'],
            'Time(s)': elapsed_time,
            'Best_Pairs': str(best_pair)
        }
        logger.info(f"Finished {data_name} with ARI: {result['ARI']:.4f} in {elapsed_time:.2f}s")
        return result
    else:
        logger.warning(f"Could not get valid clustering result for {data_name}.")
        return None




# Ensure stability for multiprocessing on Windows or macOS
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_opn(opn):
    """Ensure all inputs are converted to valid OPNs objects and handle symbols as defined in the paper."""
    if isinstance(opn, OPNs):
        return opn
    elif isinstance(opn, (tuple, list)) and len(opn) >= 2:
        return OPNs(float(opn[0]), float(opn[1]))
    elif isinstance(opn, (tuple, list)) and len(opn) == 1:
        return OPNs(float(opn[0]), 0.0)
    elif isinstance(opn, (int, float)):
        return OPNs(float(opn), 0.0)
    elif isinstance(opn, np.ndarray):
        if opn.size >= 2:
            return OPNs(float(opn[0]), float(opn[1]))
        elif opn.size == 1:
            return OPNs(float(opn[0]), 0.0)
    logger.error(f"Cannot convert type {type(opn)} to OPNs: {opn}")
    return OPNs(0.5, 0.5)


def evaluate_clustering(y_true, clusters, centers, opns_data):
    """Evaluate clustering results (add label alignment)"""
    if clusters is None or centers is None or len(clusters) == 0:
        return {'accuracy': 0.0, 'recall': 0.0, 'f1': 0.0, 'ari': 0.0}

    if not np.issubdtype(y_true.dtype, np.number):
        try:
            y_true = LabelEncoder().fit_transform(y_true)
        except:
            return {'accuracy': 0.0, 'recall': 0.0, 'f1': 0.0, 'ari': 0.0}

    y_pred = np.zeros(len(y_true), dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            y_pred[sample_idx] = cluster_idx

    try:
        unique_clusters = np.unique(y_pred)
        label_mapping = {}

        for cluster_id in unique_clusters:
            mask = (y_pred == cluster_id)
            if np.sum(mask) == 0:
                continue
            true_labels = y_true[mask]
            if len(true_labels) > 0:
                values, counts = np.unique(true_labels, return_counts=True)
                majority_label = values[np.argmax(counts)]
                label_mapping[cluster_id] = majority_label

        aligned_labels = np.zeros_like(y_pred)
        for cluster_id, true_label in label_mapping.items():
            aligned_labels[y_pred == cluster_id] = true_label

        accuracy = accuracy_score(y_true, aligned_labels)
        recall = recall_score(y_true, aligned_labels, average='macro', zero_division=0)
        f1 = f1_score(y_true, aligned_labels, average='macro', zero_division=0)
        ari = adjusted_rand_score(y_true, y_pred)

        return {'accuracy': accuracy, 'recall': recall, 'f1': f1, 'ari': ari}
    except Exception as e:
        logger.error(f"Label alignment failed: {str(e)}")
        return {'accuracy': 0.0, 'recall': 0.0, 'f1': 0.0, 'ari': 0.0}


def process_single_pair(pair, X, K, max_iterations, tol, y_true):
    """Process a single feature pair for multiprocessing"""
    try:
        logger.debug(f"Processing feature pair: {pair}")

        pairer = OPNsPairer(pair)
        pairer.fit(X)
        opns_data = pairer.transform(X)

        validated_samples = []
        if hasattr(opns_data, 'elements'):
            for sample in opns_data.elements:
                if hasattr(sample, 'elements'):
                    sample = sample.elements
                validated_samples.append([validate_opn(opn) for opn in sample])
        else:
            validated_samples = [[validate_opn(opn) for opn in sample] for sample in opns_data]

        n_samples = len(validated_samples)
        if n_samples == 0:
            raise ValueError("Transformed data is empty")

        K = min(K, n_samples)
        centers = []
        used_indices = set()

        while len(centers) < K and len(used_indices) < n_samples:
            idx = random.choice([i for i in range(n_samples) if i not in used_indices])
            center = [validate_opn(opn) for opn in validated_samples[idx]]
            if abs(center[0].a - center[0].b) > 1e-8:
                centers.append(center)
                used_indices.add(idx)

        if len(centers) < K:
            logger.warning(f"Unable to initialize enough center points (required {K}, got {len(centers)})")
            return None

        centers = [[validate_opn(opn) for opn in center] for center in centers]
        clusters = [[] for _ in range(K)]

        for iteration in range(max_iterations):
            clusters = [[] for _ in range(K)]
            for idx in range(n_samples):
                sample = validated_samples[idx]
                distances = [opnpy.generalized_metric(sample, center) for center in centers]
                min_idx = np.argmin(distances)
                clusters[min_idx].append(idx)

            # Update centers
            new_centers = []
            empty_cluster_indices = []
            for i, cluster in enumerate(clusters):
                if cluster:
                    cluster_samples_array = opnpy.array([validated_samples[idx] for idx in cluster])
                    new_center_obj = opnpy.mean(cluster_samples_array, axis=0)
                    new_centers.append(new_center_obj.elements)
                else:
                    empty_cluster_indices.append(i)
                    new_centers.append(None)  # Temporary placeholder

            # --- [Modification 1]:Robust empty cluster handling ---
            if empty_cluster_indices:
                non_empty_clusters = [c for c in clusters if c]
                if non_empty_clusters:
                    # Find the largest cluster by sample size
                    largest_cluster = max(non_empty_clusters, key=len)

                    for i in empty_cluster_indices:
                        if len(largest_cluster) > 1:
                            # Randomly "borrow" a point from the largest cluster as the new center
                            reseed_point_idx = random.choice(largest_cluster)
                            largest_cluster.remove(reseed_point_idx)  # Remove from original cluster
                            new_centers[i] = validated_samples[reseed_point_idx]
                            logger.debug(f"Empty cluster {i} has been reinitialized by taking a point from the largest cluster.")
                        else:
                            # If the largest cluster also has only one point, randomly select one from all samples
                            reseed_point_idx = random.choice(range(n_samples))
                            new_centers[i] = validated_samples[reseed_point_idx]
                            logger.debug(f"Empty cluster {i} has been reinitialized by randomly taking a point from the dataset.")

            # Filter out any remaining None (in case all clusters are empty)
            new_centers_list = [c for c in new_centers if c is not None]

            if len(new_centers_list) < K:
                logger.warning(f"After handling empty clusters, the number of centers is still insufficient ({K}), terminating iteration.")
                break

            new_centers = [[validate_opn(opn) for opn in center] for center in new_centers_list]

            # Check for convergence
            converged = True
            if len(new_centers) != len(centers):
                converged = False
            else:
                for i in range(K):
                    if len(new_centers[i]) != len(centers[i]):
                        converged = False
                        break
                    for new, old in zip(new_centers[i], centers[i]):
                        if (abs(new.a - old.a) > tol) or (abs(new.b - old.b) > tol):
                            converged = False
                            break
                    if not converged:
                        break

            centers = new_centers

            if converged:
                break

        metrics = evaluate_clustering(y_true, clusters, centers, opns_data)
        gc.collect()

        return {
            'ari': metrics['ari'],
            'clusters': clusters,
            'centers': centers,
            'pairing': pair,
            'eval_metrics': metrics
        }

    except Exception as e:
        logger.error(f"Error processing feature pair {pair}: {str(e)}", exc_info=True)
        return None


def kmeans_opns_parallel(X, K, max_iterations=100, tol=1e-4, y_true=None, n_jobs=None):
    """Multiprocess implementation of K-means based on the OPNs generalized metric"""
    if X.shape[0] == 0: raise ValueError("Input data cannot be empty")
    if y_true is None: raise ValueError("True labels y_true are required")

    feature_k_num = X.shape[1] if X.shape[1] % 2 == 0 else X.shape[1] + 1
    all_pairs_list = list(seq_all_pairs_list2([i for i in range(feature_k_num)]))

    if not all_pairs_list:
        logger.warning("No feature pairs to process.")
        return None, None, None, None

    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1

    best_result = {'ari': -1.0, 'pairing': None, 'eval_metrics': None}

    with Pool(processes=n_jobs, maxtasksperchild=10) as pool:
        process_func = functools.partial(process_single_pair, X=X, K=K, max_iterations=max_iterations, tol=tol,
                                         y_true=y_true)

        desc = f"Evaluating feature pairs for dataset {y_true.name if hasattr(y_true, 'name') else 'current'}"
        pbar = tqdm(pool.imap_unordered(process_func, all_pairs_list, chunksize=10), total=len(all_pairs_list),
                    desc=desc)

        for result in pbar:
            if result is not None and result.get('ari', -1) > best_result['ari']:
                best_result = result
                pbar.set_postfix_str(f"Best ARI: {best_result['ari']:.4f}")

    return best_result.get('clusters'), best_result.get('centers'), best_result.get('pairing'), best_result.get(
        'eval_metrics')

