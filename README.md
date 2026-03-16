Numbers# K-means Clustering with Generalized Metrics Using OPNs

This repository contains the official source code and experiment instructions for the paper: **"K-means Clustering with Generalized Metrics Using Ordered Pair of Normalized Real Numbers"**.

This project introduces a novel K-means algorithm (OPNs-K-means) that utilizes Ordered Pairs of Normalized real numbers (OPNs) to explore data in a non-real domain. This approach enhances clustering performance by capturing latent feature relationships through a custom OPN-valued generalized metric.

> **🌟 Related Work & Latest OPNs Framework**
> For the latest version of the foundational OPNs framework and other related research, please visit the official repository: **[alvinzean/OPNs](https://github.com/alvinzean/OPNs)**.

## Core Contributions
The primary contributions of this work are:

***Expansion of the Research Field**: We pioneer the application of the OPNs framework to unsupervised clustering, extending its use beyond previous supervised learning contexts.
***Novel Core Mechanism**: We introduce a bidirectional stepwise selection algorithm to create an automatic search framework for optimal feature pairing.This addresses combinatorial challenges in high-dimensional data not covered in prior OPNs research.
***Theoretical Suitability**: We clarify that the algebraically complete system of OPNs provides a solid mathematical foundation for the K-means centroid update step, an advantage not shared by other ordered pair theories like Intuitionistic Fuzzy Sets (IFNs).

## Folder Structure
OPNs-Kmeans-Clustering/
├── data/                  # Folder for datasets (from UCI, KEEL)
├── results/               # Folder to save experiment results (e.g., CSV files, plots)
├── src/                   # Source code
│   ├── common/            # Common utility functions
│   ├── data_loader/       # Scripts for loading and preprocessing data
│   ├── opns_pairer.py     # Core logic for OPNs feature pairing strategies
│   ├── opns_kmeans_a.py   # Implementation of OPNs-K-means(a)
│   └── opns_kmeans_b.py   # Implementation of OPNs-K-means(b)
├── LICENSE                # MIT License
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── run_experiments.py     # Main script to run all experiments
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd OPNs-Kmeans-Clustering
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Datasets:**
    The experiments in the paper were conducted on 23 real-world datasets, primarily from the UCI [cite: 323, 500] and KEEL [cite: 323, 501] repositories. Please download the relevant datasets (listed in Table 1 of the paper [cite: 328]) and place them in the `data/` directory.

## How to Run Experiments

The `run_experiments.py` script is the main entry point to reproduce the results from the paper.

### Reproducing Results for a Single Dataset
To run the OPNs-K-means algorithm and all baseline models on a specific dataset (e.g., `Iris`):
```bash
python run_experiments.py --dataset Iris

Reproducing All Paper Results
To run the experiments for all 23 datasets as reported in the paper:

Bash

python run_experiments.py --all
The results (ARI, ACC, F1 scores) will be saved in the 

results/ folder.

The script will automatically select the correct algorithm variant based on the dataset's feature dimensions, as described in the paper:


OPNs-K-means(a): Pairing with No Feature Reuse, applied to datasets with fewer than 10 features (DS1-DS12).


OPNs-K-means(b): Hybrid Pairing Strategy with feature reuse and stepwise selection, applied to datasets DS1-DS23.
