import pandas as pd
import numpy as np
import time
import itertools
import os
import random
import functools

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, accuracy_score, recall_score, f1_score
from scipy.stats import mode
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from .data_loader import DatasetLoader
from .opns_pairer import OPNsPairer
from .common import opnpy
from .common.opn import OPNs
from .common.gen_pairs import seq_all_pairs_with_repeats



def run_opns_kmeans_b(X, y, k_classes, data_name):
    """
    Performs a single run of the OPNs-K-means(b) algorithm.
    This algorithm uses a hybrid strategy:
    - For data with > 5 features, it uses bidirectional stepwise selection.
    - For data with <= 5 features, it uses feature reuse (global search).

    Args:
        X (np.array): The input feature data.
        y (np.array): The true labels.
        k_classes (int): The number of classes.
        data_name (str): The name of the dataset.

    Returns:
        dict: A dictionary containing the evaluation metrics and the best feature pair; returns None if it fails.
    """
    logger = logging.getLogger(__name__)
    n_features = X.shape[1]
    y_encoded = LabelEncoder().fit_transform(y)
    
    logger.info(f"Running OPNs-K-means(b) on {data_name} (features={n_features})")
    
    start_time = time.time()
    best_scores = None
    best_feature_subset = None

    # --- Core logic: select strategy based on the number of features ---
    if n_features > 6:
        logger.info(f"Strategy: Features ({n_features}) > 6, using stepwise selection.")
        best_feature_subset, best_scores = stepwise_kmeans_selector(X, k_classes, y_true=y_encoded, run_id=1)
    else:
        logger.info(f"Strategy: Features ({n_features}) <= 6, using feature reuse (global search).")
        best_feature_subset, metrics = kmeans_opns_feature_reuse(X, k_classes, y_true=y_encoded)

        # Unify the return format for both strategies
        if metrics:
            best_scores = {
                'ARI': metrics.get('ari', -1.0),
                'Accuracy': metrics.get('accuracy', -1.0),
                'Recall': metrics.get('recall', -1.0),
                'F1': metrics.get('f1', -1.0)
            }
        else:
            best_scores = None # If no metrics are returned, set to None

    elapsed_time = time.time() - start_time

    # --- Unify processing and return results ---
    if best_scores and best_scores.get('ARI', -1.0) > -1.0:
        result = {
            'ARI': best_scores['ARI'],
            'Accuracy': best_scores['Accuracy'],
            'F1': best_scores['F1'],
            'Time(s)': elapsed_time,
            'Best_Pairs': str(best_feature_subset)
        }
        logger.info(f"Finished {data_name} with ARI: {result['ARI']:.4f} in {elapsed_time:.2f}s")
        return result
    else:
        logger.warning(f"Could not get valid clustering result for {data_name}.")
        return None
    

# Configure logging
logging.basicConfig(level=logging.INFO)
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


# ==============================================================================
# Stepwise Selection Strategy (from stepwise_new.py)
# ==============================================================================

def run_kmeans_on_subset(X, K, y_true, opns_feature_pairs, max_iterations=100, tol=1e-4):
    """
    Receives a subset of OPNs feature pairs, runs K-Means on this subset, 
    and returns a dictionary with multiple evaluation metrics.
    """
    # # Instantiate OPNssPairer and pass in the current feature pairing scheme
    pairer = OPNsPairer(pair=opns_feature_pairs)

    # Use fit_transform to complete configuration and transformation in one step
    op_array = pairer.fit_transform(X)
    opns_data = op_array.elements  # 获取转换后的 OPNs 数据列表

    # The subsequent K-Means logic remains unchanged
    # Initialize centroids
    initial_indices = np.random.choice(len(opns_data), K, replace=False)
    centers = [opns_data[i] for i in initial_indices]

    cluster_indices = np.zeros(len(opns_data), dtype=int)

    for _ in range(max_iterations):
        clusters = [[] for _ in range(K)]

        # Assignment step
        for i, sample in enumerate(opns_data):
            distances = [opnpy.generalized_metric(sample, center) for center in centers]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)
            cluster_indices[i] = cluster_idx

        # Update centroids
        new_centers = []
        for i, cluster_idx_list in enumerate(clusters):
            if cluster_idx_list:
                # Get all samples in the current cluster from opns_data
                cluster_samples = [opns_data[idx] for idx in cluster_idx_list]
                cluster_array = opnpy.array(cluster_samples)
                new_center_array = opnpy.mean(cluster_array, axis=0)
                new_centers.append(new_center_array.elements)
            else:
                # If the cluster is empty, keep the original centroid unchanged
                new_centers.append(centers[i])

        total_movement = OPNs(0, 0)
        for old_center, new_center in zip(centers, new_centers):
            total_movement += opnpy.generalized_metric(old_center, new_center)
        # Compare the values
        if abs(total_movement.a) < tol and abs(total_movement.b) < tol:
            converged = True
        else:
            converged = False

        centers = new_centers
        if converged:
            break

    # --- Calculate multiple evaluation metrics ---
    ari = adjusted_rand_score(y_true, cluster_indices)

    y_pred_mapped = np.zeros_like(cluster_indices)
    for cluster_id in range(K):
        mask = (cluster_indices == cluster_id)
        if np.any(mask):
            true_label_for_cluster = mode(y_true[mask])[0]
            y_pred_mapped[mask] = true_label_for_cluster

    accuracy = accuracy_score(y_true, y_pred_mapped)
    recall = recall_score(y_true, y_pred_mapped, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_mapped, average='macro', zero_division=0)

    return {'ARI': ari, 'Accuracy': accuracy, 'Recall': recall, 'F1': f1}


def stepwise_kmeans_selector(X, K, y_true, run_id):
    """
    Implement a forward-backward stepwise selection to find the optimal feature pair combination.
    Introduce a visited_states set: Before the while loop starts, create a set called visited_states to store all evaluated feature combination states.
    State transition and detection: At the beginning of each round of the loop, convert the current selected_pairs (a list) into a hashable, order-independent frozenset.
    If the current state has already appeared in visited_states, it means the algorithm has entered a loop, and the program will print a warning and immediately terminate the loop.
    If it is a new state, it will be added to visited_states, and the execution will continue.
    """
    n_features = X.shape[1]
    all_simple_pairs = list(itertools.combinations(range(n_features), 2))

    selected_pairs = []
    best_overall_scores = {'ARI': -1.0, 'Accuracy': -1.0, 'Recall': -1.0, 'F1': -1.0}
    visited_states = set()
    logger.info(f"Starting bidirectional stepwise selection, total candidate pairs: {len(all_simple_pairs)}")

    while True:
        # Convert the current feature pair combination into a hashable, order-independent object
        current_state = frozenset(tuple(sorted(p)) for p in selected_pairs)
        if current_state in visited_states:
            logger.warning("Selection Warning: State cycle detected. Terminating early to prevent an infinite loop.")
            break
        visited_states.add(current_state)
        made_change_in_pass = False

        # --- Forward Step ---
        best_scores_in_forward_step = best_overall_scores
        best_pair_to_add = None

        candidate_pool = [p for p in all_simple_pairs if p not in selected_pairs]
        if candidate_pool:
            pbar_fwd = tqdm(candidate_pool, desc=f"Forward (Current ARI: {best_overall_scores['ARI']:.4f})", leave=False)
            for candidate_pair in pbar_fwd:
                temp_subset = selected_pairs + [candidate_pair]
                current_scores = run_kmeans_on_subset(X, K, y_true, temp_subset)
                if current_scores['ARI'] > best_scores_in_forward_step['ARI']:
                    best_scores_in_forward_step = current_scores
                    best_pair_to_add = candidate_pair
                    pbar_fwd.set_postfix_str(f"Best ARI this step: {current_scores['ARI']:.4f}")

        if best_pair_to_add:
            selected_pairs.append(best_pair_to_add)
            best_overall_scores = best_scores_in_forward_step
            made_change_in_pass = True
            logger.info(f"Forward: Adopting feature pair {best_pair_to_add},new best ARI: {best_overall_scores['ARI']:.4f}")

        # --- Backward Step ---
        best_scores_in_backward_step = best_overall_scores
        pair_to_remove = None

        if len(selected_pairs) > 1:
            pbar_bwd = tqdm(selected_pairs, desc=f"Backward (Current ARI: {best_overall_scores['ARI']:.4f})", leave=False)
            for p in pbar_bwd:
                temp_subset = [sp for sp in selected_pairs if sp != p]
                current_scores = run_kmeans_on_subset(X, K, y_true, temp_subset)
                if current_scores['ARI'] > best_scores_in_backward_step['ARI']:
                    best_scores_in_backward_step = current_scores
                    pair_to_remove = p
                    pbar_bwd.set_postfix_str(f"If {p} removed, ARI improves to: {current_scores['ARI']:.4f}")

        if pair_to_remove:
            selected_pairs.remove(pair_to_remove)
            best_overall_scores = best_scores_in_backward_step
            made_change_in_pass = True
            logger.info(f"Backward: Removing feature pair {pair_to_remove}, new best ARI:  {best_overall_scores['ARI']:.4f}")

        # If a full forward-backward pass makes no changes, exit the loop
        if not made_change_in_pass:
            logger.info("Selection finished: No improvement found in a full forward-backward pass.")
            break

    return selected_pairs, best_overall_scores


# ==============================================================================
# 特征复用策略
# ==============================================================================

def evaluate_clustering(y_true, clusters, centers, opns_data):
    """Evaluate clustering results (with label alignment)"""
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
                majority_label = mode(true_labels).mode
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
            logger.warning(f"Unable to initialize enough centers (required {K}, got {len(centers)})")
            return None

        centers = [[validate_opn(opn) for opn in center] for center in centers]

        for iteration in range(max_iterations):
            clusters = [[] for _ in range(K)]

            for idx in range(n_samples):
                sample = validated_samples[idx]
                distances = []

                try:
                    sample = [validate_opn(x) for x in sample]
                except:
                    logger.warning(f"Sample {idx} has invalid format")
                    continue

                for center in centers:
                    try:
                        if not center:
                            distances.append(OPNs(float('inf'), float('inf')))
                            continue

                        center = [validate_opn(x) for x in center]
                        dist = opnpy.generalized_metric(sample, center)
                        if dist.a + dist.b > 0:
                            dist = OPNs(float('inf'), float('inf'))
                        distances.append(dist)
                    except Exception as e:
                        logger.warning(f"Distance calculation failed: {str(e)}")
                        distances.append(OPNs(float('inf'), float('inf')))

                min_idx = np.argmin(distances)
                if min_idx < len(clusters):
                    clusters[min_idx].append(idx)

            new_centers = []
            empty_clusters = []

            for i, cluster in enumerate(clusters):
                if cluster:
                    cluster_samples = [validated_samples[idx] for idx in cluster]
                    # Convert the list of samples in the cluster to an opnpy.ndarray
                    cluster_array = opnpy.array(cluster_samples)

                    # Calculate the mean along axis 0 (column-wise) using the package function
                    new_center_array = opnpy.mean(cluster_array, axis=0)

                    # The result is an opnpy.ndarray, so extract its elements
                    new_center = new_center_array.elements
                else:
                    empty_clusters.append(i)
                    new_center = None

                new_centers.append(new_center)

            if empty_clusters:
                non_empty_samples = [idx for cluster in clusters for idx in cluster if cluster]
                if non_empty_samples:
                    for i in empty_clusters:
                        split_point = random.choice(non_empty_samples)
                        new_centers[i] = [validate_opn(opn) for opn in validated_samples[split_point]]
                        for c in clusters:
                            if split_point in c:
                                c.remove(split_point)
                                break
                        clusters[i].append(split_point)
                else:
                    logger.warning("All clusters are empty, reinitializing centers")
                    centers = []
                    used_indices = set()
                    while len(centers) < K and len(used_indices) < n_samples:
                        idx = random.choice([i for i in range(n_samples) if i not in used_indices])
                        center = [validate_opn(opn) for opn in validated_samples[idx]]
                        if abs(center[0].a - center[0].b) > 1e-8:
                            centers.append(center)
                            used_indices.add(idx)
                    if len(centers) < K:
                        break

            new_centers = [[validate_opn(opn) for opn in center] if center is not None else None for center in
                           new_centers]

            converged = True
            for i in range(K):
                if new_centers[i] is None:
                    converged = False
                    break
                for new, old in zip(new_centers[i], centers[i]):
                    if (abs(new.a - old.a) > tol) or (abs(new.b - old.b) > tol):
                        converged = False
                        break
                if not converged:
                    break

            centers = [c for c in new_centers if c is not None]
            if len(centers) < K:
                logger.warning(f"Number of centers is insufficient {K}, terminating iteration")
                break

            if converged:
                logger.debug(f"Converged after {iteration} iterations")
                break

        try:
            y_pred = np.zeros(len(y_true), dtype=int)
            for cluster_idx, cluster in enumerate(clusters):
                for sample_idx in cluster:
                    y_pred[sample_idx] = cluster_idx

            metrics = evaluate_clustering(y_true, clusters, centers, opns_data)

            return {
                'ari': metrics['ari'],
                'clusters': clusters,
                'centers': centers,
                'pairing': pair,
                'eval_metrics': metrics
            }
        except Exception as e:
            logger.error(f"Metric calculation failed: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error processing feature pair {pair}: {str(e)}", exc_info=True)
        return None


def kmeans_opns_feature_reuse(X, K, max_iterations=100, tol=1e-4, y_true=None, n_jobs=None):
    """K-means multiprocessing implementation based on OPNs generalized metrics (with progress bar and memory optimization)"""
    if X.shape[0] == 0:
        raise ValueError("Input data cannot be empty")
    if y_true is None:
        raise ValueError("True labels y_true are required")

    feature_k_num = X.shape[1] if X.shape[1] % 2 == 0 else X.shape[1] + 1
    all_pairs_list = list(seq_all_pairs_with_repeats([i for i in range(feature_k_num)]))

    if not all_pairs_list:
        logger.warning("No feature pairs to process.")
        return None, None

    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1

    best_result = {'ari': -1.0, 'pairing': None, 'eval_metrics': None}

    with Pool(processes=n_jobs) as pool:
        process_func = functools.partial(
            process_single_pair,
            X=X, K=K, max_iterations=max_iterations, tol=tol, y_true=y_true
        )
        # Use the length of all_pairs_list as total.
        # No longer use list() to collect all results at once to avoid memory overflow.
        desc = f"Evaluating feature pairs (total {len(all_pairs_list)})"
        pbar = tqdm(pool.imap_unordered(process_func, all_pairs_list), total=len(all_pairs_list), desc=desc)

        for result in pbar:
            if result is not None and result['ari'] > best_result['ari']:
                best_result = result
                # Update the progress bar's postfix information to show the current best ARI
                pbar.set_postfix_str(f"Best ARI: {best_result['ari']:.4f}")

    best_pairing = best_result.get('pairing')
    eval_metrics = best_result.get('eval_metrics')

    return best_pairing, eval_metrics


