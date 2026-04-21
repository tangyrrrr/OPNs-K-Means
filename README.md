# K-means Clustering with Generalized Metrics Using OPNs

This repository contains the code, baseline implementations, and plotting/statistical scripts for the paper:

**K-means Clustering with Generalized Metrics Using Ordered Pair of Normalized Real Numbers**

The repository is provided to support the reproducibility of the experimental results reported in the manuscript, including the main comparison tables and figures.

## Overview

This project studies an OPNs-based extension of K-means, where clustering is performed in a generalized-metric space induced by **Ordered Pairs of Normalized real numbers (OPNs)**.

The main practical focus of the method is to model **explicit and interpretable second-order feature interactions** while retaining competitive clustering performance.

For the broader OPNs framework and related work, please refer to:  
[alvinzean/OPNs](https://github.com/alvinzean/OPNs)

## Repository Structure

```text
.
├── OPNs-Kmeans-Clustering/          # Main implementation of OPNs-K-means and experiment runner
├── baseline/                        # Baseline algorithms used for comparison
│   ├── Badeline_Cos.py              # Cosine-distance K-means baseline
│   ├── Baseline_E.py                # Euclidean-distance K-means baseline
│   ├── Baseline_M.py                # Manhattan-distance K-means baseline
│   ├── Baseline_T_kmeans.py         # Tanimoto-based K-means baseline
│   └── kkm.py                       # Kernel K-means baselines
├── Plot_CD_fig6/
│   └── result_Statistical/          # Statistical resources for Fig. 6
├── Plot_pairs_fig7/                 # Scripts/resources for Fig. 7
├── Plot_time_fig8/                  # Scripts/resources for Fig. 8
├── Plot_feature_del_fig9/           # Scripts/resources for Fig. 9
├── Plot_stability_fig10/            # Scripts/resources for Fig. 10
├── TDEC-main.zip                    # Archived deep clustering baseline
├── idc-main.zip                     # Archived deep clustering baseline
├── zeus.zip                         # Archived deep clustering baseline
└── README.md
```

## Method Variants

The paper evaluates two configurations of the proposed method:

- **OPNs-K-means(a) (Pairing with No Feature Reuse):**  
  This variant models interpretable second-order feature interactions through non-redundant feature pairing. It is applied to datasets with fewer than 10 features (DS1–DS12).

- **OPNs-K-means(b) (Hybrid Pairing Strategy):**  
  This variant is applied to all datasets. It uses relational pairing with feature reuse for low-dimensional data (`d <= 5`) and dynamically switches to bidirectional stepwise selection for higher-dimensional data (`d > 5`) to control combinatorial complexity.

## Requirements

The main implementation uses Python. According to the manuscript, the experiments rely primarily on:

- `scikit-learn`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

Please install the dependencies listed in the `requirements.txt` file inside `OPNs-Kmeans-Clustering/`.

Some baseline or plotting scripts may require additional packages.

## Experimental Environment

According to the manuscript:

- Traditional, kernel, and OPNs-K-means algorithms were executed on a **Windows 10** workstation with an **Intel i5-1135G7 CPU** and **16 GB RAM**.
- Deep clustering baselines were executed on an **Ubuntu 22.04** server with a **32-core AMD EPYC 9754 CPU** and **40 GB RAM**.
- Multiprocessing was used for pair evaluations in the OPNs-based experiments.

## Setup

Clone the repository:

```bash
git clone https://github.com/tangyrrrr/OPNs-K-Means.git
cd OPNs-K-Means
```

Enter the main implementation folder:

```bash
cd OPNs-Kmeans-Clustering
```

Create a virtual environment if needed.

**Linux / macOS**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The experiments use **23 real-world benchmark datasets**, primarily from the **UCI Machine Learning Repository** and the **KEEL Dataset Repository**.

Please download the datasets reported in Table 1 of the paper and place them in the directory required by the main implementation.

Before running the experiments, make sure that:

- all datasets reported in the manuscript are available locally,
- filenames match those expected by the loader scripts,
- preprocessing is consistent with the manuscript.

## Running the Main Experiments

The main experiment runner is located in `OPNs-Kmeans-Clustering/`.

Run a single dataset:

```bash
python run_experiments.py --dataset Iris
```

Run all datasets:

```bash
python run_experiments.py --all
```

The generated outputs, including clustering scores and aggregated results, are saved under:

```text
OPNs-Kmeans-Clustering/results/
```

## Baseline Algorithms

The comparison methods used in the paper are included in the `baseline/` folder:

- `Baseline_E.py`: Euclidean-distance K-means
- `Baseline_M.py`: Manhattan-distance K-means
- `Badeline_Cos.py`: Cosine-distance K-means
- `Baseline_T_kmeans.py`: Tanimoto-based K-means
- `kkm.py`: Kernel K-means variants

These scripts provide the traditional distance-based and kernel-based baselines used in the comparative experiments.

Archived implementations for the deep clustering baselines are also included:

- `TDEC-main.zip`
- `idc-main.zip`
- `zeus.zip`

If these baselines need to be rerun directly, unzip them first and follow their corresponding instructions.

## Baseline Configuration Summary

To match the manuscript as closely as possible, the baseline configurations are summarized below.

### Traditional distance-based baselines
For EDK, MDK, CDK, and TDK:

- K-means++ initialization was used,
- `n_init = 5`,
- the best run was selected according to the corresponding internal criterion:
  - minimum distance for EDK and MDK,
  - maximum similarity for CDK and TDK.

### Kernel K-means baselines
For KKM(p), KKM(l), and KKM(r):

- KKM(p) uses a **degree-2 polynomial kernel** with `coef0 = 1.0`,
- KKM(r) uses an **RBF kernel** with `gamma` tuned around the median-heuristic estimate:
  - `gamma × {0.5, 1.0, 2.0}`,
- K-means++ initialization was used,
- `n_init = 5`,
- optimization stopped when cluster assignments became stable or when the maximum number of iterations (`100`) was reached.

### Deep clustering baselines
For TDEC, ZEUS, and IDC:

- implementations follow the experimental settings in the original papers,
- Adam or AdamW was used as the optimizer,
- original learning rates and batch sizes were used:
  - IDC: learning rate `1e-3`, batch size `256`
  - TDEC: learning rate `1e-4`, batch size `128/384`
  - ZEUS: learning rate `2e-5`
- training terminated either by early stopping with threshold `δ = 1e-3` or by the preset maximum number of epochs:
  - IDC: `1000`
  - TDEC: `30`
  - ZEUS: `300`

### Repeated runs and seed schedules
For fair statistical comparison:

- all kernel and deep baselines were evaluated over **10 repeated runs**,
- KKM used seeds `{42, 43, ..., 51}`,
- TDEC and IDC used seeds `{0, 1, ..., 9}`,
- ZEUS used seeds `{2026, 2027, ..., 2035}`.

## Additional Experimental Settings

According to the manuscript:

- all K-means variants used a maximum of **100 iterations**,
- the number of clusters `K` was set equal to the number of ground-truth classes,
- traditional baselines used a convergence tolerance of **1e-4**,
- for OPNs-K-means, convergence required the absolute variations in both centroid components (`μ` and `ν`) to simultaneously fall below **1e-4**.

## Reproducibility: Regenerating the Main Tables and Figures

This repository contains the main OPNs-Kmeans implementation, baseline algorithms used for comparison, and dedicated plotting/statistical scripts for reproducing the principal results reported in the paper.

### Step 1: Run the clustering experiments

Generate the main experimental outputs:

```bash
cd OPNs-Kmeans-Clustering
python run_experiments.py --all
```

These outputs are used to produce the paper tables and as inputs to the plotting/statistical scripts.

### Step 2: Regenerate the main figures

Use the corresponding folders to regenerate the figures reported in the paper:

- **Fig. 6**: `Plot_CD_fig6/`  
  Statistical comparison / Critical Difference analysis.

- **Fig. 7**: `Plot_pairs_fig7/`  
  Feature-pair interaction analysis and visualization.  
  Known script: `pairs.py`

- **Fig. 8**: `Plot_time_fig8/`  
  Runtime-performance / computational cost analysis.  
  Known script: `Time_Performance_Tradeoff.py`

- **Fig. 9**: `Plot_feature_del_fig9/`  
  Feature deletion / selected OPNs-pair dimensionality analysis.  
  Known script: `feature_del_plot.py`

- **Fig. 10**: `Plot_stability_fig10/`  
  Stability analysis under random initialization.

Please run the plotting/statistical script contained in each corresponding folder using the result files generated in Step 1.

## Interpretation of the Method

The implementation is designed to study OPNs-K-means as a clustering framework that combines:

- a generalized metric in the OPNs domain,
- explicit second-order feature interaction modeling,
- interpretable feature-pair selection.

The method is intended as a transparent alternative to purely implicit kernel and deep clustering representations.

## License

This project is released under the license specified in the repository.

## Citation

If you use this repository in your research, please cite the corresponding paper.


## Contact

For questions about the paper or the code, please contact the corresponding authors via the contact information provided in the manuscript.
