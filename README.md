# K-means Clustering with Generalized Metrics Using OPNs

This repository contains the code, baseline implementations, and plotting/statistical scripts for the paper:

**K-means Clustering with Generalized Metrics Using Ordered Pair of Normalized Real Numbers**

The repository is provided to support the reproducibility of the experimental results reported in the manuscript, including the main comparison tables and figures.

## Overview

This project studies an OPNs-based extension of K-means, where clustering is performed in a generalized-metric space induced by **Ordered Pairs of Normalized real numbers (OPNs)**.

The practical focus of the method is to model **explicit and interpretable second-order feature interactions** while retaining competitive clustering performance.

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

> **Note:** The filename `Badeline_Cos.py` is written here to match the current repository content. If it is later renamed to `Baseline_Cos.py`, please update this README accordingly.

## Requirements

The main implementation uses Python. Please install the dependencies listed in the `requirements.txt` file inside `OPNs-Kmeans-Clustering/`.

Typical packages include:

- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `matplotlib`

Some baseline or plotting scripts may require additional packages.

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

The experiments use **23 real-world datasets**, primarily from the **UCI Machine Learning Repository** and the **KEEL Dataset Repository**.

Please download the datasets used in the paper and place them in the directory required by the main implementation.

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

The generated outputs, including clustering scores and aggregated results, are typically saved under:

```text
OPNs-Kmeans-Clustering/results/
```

## Baseline Algorithms

The conventional comparison methods used in the paper are included in the `baseline/` folder:

- `Baseline_E.py`: Euclidean-distance K-means
- `Baseline_M.py`: Manhattan-distance K-means
- `Badeline_Cos.py`: Cosine-distance K-means
- `Baseline_T_kmeans.py`: Tanimoto-based K-means
- `kkm.py`: Kernel K-means variants

These scripts provide the distance-based and kernel-based baselines used in the comparative experiments.

## Reproducibility: Regenerating the Main Tables and Figures

This repository contains the main OPNs-K-means implementation, baseline algorithms used for comparison, and dedicated plotting/statistical scripts for reproducing the principal results reported in the paper.

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
- **Fig. 7**: `Plot_pairs_fig7/`
- **Fig. 8**: `Plot_time_fig8/`
- **Fig. 9**: `Plot_feature_del_fig9/`
- **Fig. 10**: `Plot_stability_fig10/`

Please run the plotting/statistical script contained in each folder using the result files generated in Step 1.

### Step 3: Deep clustering baselines

Archived implementations for the deep clustering baselines are also included:

- `TDEC-main.zip`
- `idc-main.zip`
- `zeus.zip`

If these baselines need to be rerun directly, unzip them first and follow their corresponding instructions.

## Experimental Notes

To reproduce results as closely as possible to those reported in the manuscript, please use:

- the package versions listed in `requirements.txt`,
- the stopping criteria described in the paper,
- the same random-seed schedule used in the released experiments,
- the same preprocessing conventions as in the manuscript.

Please refer to the paper for details on hardware/software settings, iteration limits, convergence thresholds, and statistical analysis.



## Contact

For questions about the paper or the code, please contact the corresponding authors via the contact information provided in the manuscript.
