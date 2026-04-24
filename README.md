# ACE vs MACE Comparison Benchmark

## Context
This repository contains an empirical comparison between the **ACE (Atomic Cluster Expansion)** and **MACE (Message Passing ACE)** interatomic potential frameworks. The goal is to evaluate the tradeoff between the computational speed of the strictly local ACE model and the increased accuracy provided by the multi-layer graph neural network approach in MACE. 

## Experimental Setup
To ensure a fair mathematical and computational comparison, both models are trained on the exact same dataset subset using identical hyperparameters:

- **Cutoff Radius**: 5.0 Å
- **ACE Body Order (Correlation order $\nu$)**: 2 (3-body interactions)
- **MACE Blocks**: 2 layers of message passing
- **Maximum Angular Momentum ($l_{max}$)**: 2
- **Batch Size**: 32 (or adjusted based on VRAM)
- **Optimizer**: AdamW
- **LR Schedule**: ReduceLROnPlateau

### Metrics Tracked
1. **Accuracy**: Energy MAE (meV/atom) and Force MAE (meV/Å).
2. **Computational Speed**: Wall-clock time per epoch (seconds/epoch).
3. **Data Efficiency & Total Time**: Total time to convergence.

## Directory Structure
```
ace-vs-mace-benchmark/
├── README.md                      # Comprehensive guide on how to run the project
├── requirements.txt               # Dependencies
├── data/                          # Cu Molecular Dynamics (MD) trajectory dataset
├── src/                           # Modular Python scripts for datasets, model wrappers, and training
│   ├── dataset.py                 # PyTorch DataLoader for CIFs
│   ├── models/                    # Fully self-contained local ACE and MACE implementations
│   └── trainer.py                 # Unified training loop
└── notebooks/                     # Core workflow
    ├── 01_Data_Preparation.ipynb  
    ├── 02_ACE_Training.ipynb      
    ├── 03_MACE_Training.ipynb     
    └── 04_Results_Comparison.ipynb 
```

## Detailed Setup & Execution Guide

Follow these steps to set up the environment and run the benchmark from scratch. 

*Note: All necessary model code for ACE and MACE has been imported directly into `src/models/` for this benchmarking repository to be fully self-contained. No external repositories are needed!*

### 1. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

**Using Python `venv` (Windows/PowerShell):**
```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# Install required dependencies
pip install -r requirements.txt
```

*(If you are on Linux/macOS, activate using `source venv/bin/activate` instead).*

### 2. Launching the Benchmark

Once the dependencies are installed, start the Jupyter Notebook server:

```powershell
jupyter notebook
```

This will open your default web browser. Navigate to the `notebooks/` directory to begin the workflow.

### 3. Notebook Workflow

Run the notebooks sequentially to execute the benchmark:

- **`01_Data_Preparation.ipynb`**: Run this first. It will generate (or fetch) the structural data, assign energies and forces, and securely split it into 1000 training, 200 validation, and 200 testing samples. The outputs are saved as `.extxyz` files in the `data/` directory.
- **`02_ACE_Training.ipynb`**: This notebook initializes the local ACE model and executes the `BenchmarkTrainer`. It will log validation metrics for every epoch and save the final metrics to `data/ace_metrics.csv`.
- **`03_MACE_Training.ipynb`**: This notebook initializes the message-passing MACE model and runs it through the exact same `BenchmarkTrainer`. It saves the final metrics to `data/mace_metrics.csv`.
- **`04_Results_Comparison.ipynb`**: Run this last. It reads the CSV logs from both models and generates beautiful side-by-side plots for Energy MAE, Force MAE, and Time per epoch.

### 🚀 Performance Tips
- **Graph Caching**: The dataset pre-computes periodic neighbor graphs once at startup. You will see a progress bar during initialization; this saves minutes of redundant computation during training.
- **CPU Benchmarking**: We use `l_max=2` by default. Equivariant models (MACE) are computationally intensive; for 1000 structures on a modern CPU, expect ~30-40s per epoch.

### Hardware Notes
If you are running this on a GPU-enabled machine, PyTorch will automatically utilize CUDA to speed up the tensor operations. Due to the message-passing overhead, you should clearly see MACE taking longer per epoch while achieving lower validation errors compared to the strictly local ACE!
