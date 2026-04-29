# ACE vs MACE Comparison Benchmark

## Context
This repository contains an empirical comparison between the **ACE (Atomic Cluster Expansion)** and **MACE (Message Passing ACE)** interatomic potential frameworks. The goal is to evaluate the tradeoff between the computational speed of the strictly local ACE model and the increased accuracy provided by the multi-layer graph neural network approach in MACE. 

## Experimental Setup
Both models are trained on the exact same train/validation/test dataset split.

The benchmark uses a shared training setup and model-specific architecture parameters:

- **Shared Training Setup**
    - Cutoff Radius: 5.0 A
    - Radial Basis Size: 8
    - Maximum Angular Momentum (l_max): 2
    - Batch Size: 128
    - Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
    - LR Schedule: ReduceLROnPlateau (factor=0.5, patience=5)
    - Trainer Schedule: max_epochs=50, early_stopping_patience=10
    - Loss Weights: energy_weight=1.0, force_weight=100.0

- **Model-Specific Architecture**
    - ACE: hidden_dim=32
    - MACE: node_dim=16, num_blocks=2

### Metrics Tracked
1. **Training/Validation Accuracy**: Energy MAE (meV/atom) and Force MAE (meV/A).
2. **Held-Out Test Accuracy**: Test Energy MAE and Test Force MAE from the test split.
3. **Computational Speed**: Wall-clock time per epoch and total training time.
4. **Optimization Metrics**: Weighted train/validation losses saved per epoch.

## Directory Structure
```
ace-vs-mace-benchmark/
├── README.md                      # Comprehensive guide on how to run the project
├── requirements.txt               # Dependencies
├── data/                          # Cu Molecular Dynamics (MD) trajectory dataset
├── src/                           # Modular Python scripts for datasets, model wrappers, and training
│   ├── dataset.py                 # PyTorch Geometric dataset for extxyz trajectories
│   ├── models/                    # Fully self-contained local ACE and MACE implementations
│   └── trainer.py                 # Unified training loop
└── notebooks/                     # Core workflow
    ├── 01_Data_Preparation.ipynb
    ├── 02_ACE_Training.ipynb
    ├── 03_MACE_Training.ipynb
    ├── 04_Inference_Benchmark.ipynb
    └── 05_Results_Comparison.ipynb
```

## Detailed Setup & Execution Guide

Follow these steps to set up the environment and run the benchmark from scratch. 

*Note: All necessary model code for ACE and MACE has been imported directly into `src/models/` for this benchmarking repository to be fully self-contained. No external repositories are needed!*

### 1. Environment Setup

It is highly recommended to use **`uv`**, an extremely fast Python package manager, to manage the environment. `uv` will automatically handle fetching the correct Python version needed for PyTorch CUDA support.

**Using uv (Windows/macOS/Linux):**
```powershell
# 1. Create a virtual environment using Python 3.11 (uv will download it automatically if missing)
uv venv --python 3.11

# 2. Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# 3. Install PyTorch with CUDA and all other dependencies
uv pip install -r requirements.txt
```

### 2. Launching the Benchmark

Once the dependencies are installed, start the Jupyter Notebook server:

```powershell
jupyter notebook
```

This will open your default web browser. Navigate to the `notebooks/` directory to begin the workflow.

### 3. Notebook Workflow

Run the notebooks sequentially to execute the benchmark:

- **`01_Data_Preparation.ipynb`**: Run this first. It generates Cu MD trajectory data with energies and forces and splits it into multiple training fractions (10%, 40%, 70%, 100%), 200 validation, and 200 test structures in `.extxyz` format.
- **`02_ACE_Training.ipynb`**: Trains ACE sequentially on the 10%, 40%, 70%, and 100% data fractions to evaluate data scaling. Writes models and metrics to `data/`.
- **`03_MACE_Training.ipynb`**: Trains MACE sequentially on the 10%, 40%, 70%, and 100% data fractions to evaluate data scaling. Writes models and metrics to `data/`.
- **`04_Inference_Benchmark.ipynb`**: Measures the forward-pass inference overhead per molecule for both ACE and MACE, outputting `data/inference_metrics.csv`.
- **`05_Results_Comparison.ipynb`**: Reads all training and inference outputs to generate the final plots: Baseline Error Curves, Data Volume Scaling (Experiment A), and Computational Cost vs Accuracy (Experiment B).

### 🚀 Performance Tips
- **Graph Caching**: The dataset pre-computes periodic neighbor graphs once at startup. You will see a progress bar during initialization; this saves minutes of redundant computation during training.
- **Runtime Expectations**: MACE is significantly more expensive per epoch than ACE. Use the generated `time` columns in metrics CSVs as the canonical measurement for your hardware.

### Hardware Notes
If you are running on a GPU-enabled machine, PyTorch can utilize CUDA to accelerate tensor operations. Because MACE includes message-passing blocks, it is expected to run slower per epoch than ACE at similar dataset sizes.
