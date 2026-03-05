import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 从src包中导入您的模块 ---
# 确保您的data_loader可以被这样调用
from src.data_loader import DatasetLoader 
# 导入我们之前重构好的算法函数
from src.OPNs_K_means_a import run_opns_kmeans_a 
from src.OPNs_K_means_b import run_opns_kmeans_b
# 您也可以将基线模型（EDK, MDK等）封装后从这里导入
# from src.baselines import run_edk, run_mdk 

def main():
    """
    主执行函数：解析参数，循环运行实验，并保存结果。
    """
    parser = argparse.ArgumentParser(description="Run experiments for OPNs-K-means paper.")
    parser.add_argument('--dataset', type=str, help='Run on a single dataset name (e.g., iris, wine). Use lowercase.')
    parser.add_argument('--all', action='store_true', help='Run on all 23 datasets from the paper.')
    parser.add_argument('--trials', type=int, default=1, help='Number of independent runs for each dataset.')
    args = parser.parse_args()

    # --- 根据论文定义，为每个算法分配数据集 ---
    # [cite_start]OPNs-K-means(a) is for datasets with d < 10 (DS1-DS12) [cite: 349]
    # [cite_start]OPNs-K-means(b) is for datasets where d > 5 (DS6-DS23) [cite: 350]
    # 注意：数据集名称使用小写，以便于命令输入
    datasets_for_a = ['iris', 'balance-scale', 'weather', 'hayes_roth', 'phoneme', 
                      'monk-2', 'led7digit', 'appendicitis', 'ecoli', 'pima', 'cars', 'saheart']
    # datasets_for_b = ['iris', 'balance-scale', 'weather', 'hayes_roth', 'phoneme','monk-2', 'led7digit', 'appendicitis', 'ecoli', 'pima', 'cars', 'saheart', 
    #                   'heart', 'cleve', 'cleveland', 'wine', 'vowel', 'penbased', 'vehicle', 
    #                   'hepatitis', 'segment', 'sonar', 'air']
    
    datasets_for_b = ['monk-2', 'led7digit', 'appendicitis', 'ecoli']
    
    # 动态获取所有可用的数据集加载器
    # 假设您的DatasetLoader类中的加载函数都是静态方法或类方法
    all_loaders = {
        name.lower(): loader for name, loader in DatasetLoader.__dict__.items() 
        if callable(loader) and not name.startswith('__')
    }
    
    # --- 确定要运行的数据集 ---
    if args.all:
        target_datasets = sorted(list(all_loaders.keys()))
    elif args.dataset:
        target_datasets = [args.dataset.lower()]
    else:
        logger.error("Please specify a dataset with --dataset <name> or run all with --all.")
        return

    # --- 主循环：遍历数据集和运行次数 ---
    all_run_results = []
    
    for dataset_name in tqdm(target_datasets, desc="Total Progress"):
        if dataset_name not in all_loaders:
            logger.warning(f"Dataset loader for '{dataset_name}' not found. Skipping.")
            continue
            
        try:
            loader = all_loaders[dataset_name]
            X, y, _, n_features, k_classes = loader()
            if X is None or y is None:
                logger.warning(f"Failed to load data for '{dataset_name}'. Skipping.")
                continue
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            continue

        for trial in tqdm(range(1, args.trials + 1), desc=f"Dataset: {dataset_name}", leave=False):
            # 运行 OPNs-K-means(a)
            if dataset_name in datasets_for_a:
                result_a = run_opns_kmeans_a(X.copy(), y.copy(), k_classes, dataset_name)
                if result_a:
                    result_a.update({'Algorithm': 'OPNs-K-means(a)', 'Dataset': dataset_name, 'Trial': trial})
                    all_run_results.append(result_a)

            # 运行 OPNs-K-means(b)
            if dataset_name in datasets_for_b:
                result_b = run_opns_kmeans_b(X.copy(), y.copy(), k_classes, dataset_name)
                if result_b:
                    result_b.update({'Algorithm': 'OPNs-K-means(b)', 'Dataset': dataset_name, 'Trial': trial})
                    all_run_results.append(result_b)

            # 您可以在此处添加运行基线模型（EDK, MDK等）的代码...

    if not all_run_results:
        logger.warning("No results were generated. Please check your setup and data.")
        return

    # --- 结果处理与保存 ---
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. 保存所有单次运行的详细结果
    results_df = pd.DataFrame(all_run_results)
    detailed_path = 'results/all_trials_detailed_results.csv'
    results_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    logger.info(f"\nDetailed results saved to '{detailed_path}'")

    # 2. 计算并保存最终的统计摘要（平均值 ± 标准差）
    summary_df = results_df.groupby(['Dataset', 'Algorithm']).agg(
        # 计算均值和标准差
        Avg_ARI=('ARI', 'mean'), Std_ARI=('ARI', 'std'),
        Avg_ACC=('Accuracy', 'mean'), Std_ACC=('Accuracy', 'std'),
        Avg_F1=('F1', 'mean'), Std_F1=('F1', 'std'),
        Avg_Time=('Time(s)', 'mean'), Std_Time=('Time(s)', 'std')
    ).reset_index()

    # 格式化输出
    summary_df['ARI'] = summary_df.apply(lambda row: f"{row['Avg_ARI']:.4f} ± {row['Std_ARI']:.4f}", axis=1)
    summary_df['ACC'] = summary_df.apply(lambda row: f"{row['Avg_ACC']:.4f} ± {row['Std_ACC']:.4f}", axis=1)
    summary_df['F1-Score'] = summary_df.apply(lambda row: f"{row['Avg_F1']:.4f} ± {row['Std_F1']:.4f}", axis=1)
    summary_df['Time(s)'] = summary_df.apply(lambda row: f"{row['Avg_Time']:.2f} ± {row['Std_Time']:.2f}", axis=1)
    
    # 选择最终要显示的列
    final_columns = ['Dataset', 'Algorithm', 'ARI', 'ACC', 'F1-Score', 'Time(s)']
    summary_final_df = summary_df[final_columns]

    summary_path = 'results/final_summary_statistics.csv'
    summary_final_df.to_csv(summary_path, index=False, encoding='utf-8-sig')    
    logger.info(f"Final summary saved to '{summary_path}'")
    
    logger.info("\n" + "="*25 + " FINAL SUMMARY " + "="*25)
    print(summary_final_df.to_string(index=False))


if __name__ == '__main__':
    main()