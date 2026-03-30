# State Estimation in Spatiotemporal Chaos via Low-Rank StatFEM

This repository contains the numerical implementation and experimental framework for the **Low-Rank Extended Rauch-Tung-Striebel Smoother (LR-ExRTSS)** within a **Statistical Finite Element (StatFEM)** context. 

The project demonstrates the reconstruction of chaotic trajectories in the **2D Anisotropic Kuramoto-Sivashinsky Equation (KSE)** under severe structural misspecification.

## Overview

The core contribution of this work is a computationally efficient, grid-independent retrospective assimilation scheme. We demonstrate that the LR-ExRTSS effectively "re-inflates" physical instabilities that are suppressed by an over-stabilised, misspecified predictive model. 

### Key Features
* **LR-ExRTSS:** A low-rank Bayesian smoother utilizing a spectral Hilbert space approximation of Gaussian processes.
* **2D Anisotropic KSE:** Numerical solvers for the data-generating process (DGP) and the misspecified predictive model ($\kappa$ mismatch).
* **Spectral Analysis:** Tools for analysing the effective rank and unstable-neutral subspaces of chaotic state estimates.

## Installation

The project uses `conda` to manage dependencies and a `Makefile` to automate the environment setup. 

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ks-smoother.git
   cd ks-smoother
   ```

2. **Configure the environment:**
   This command creates the `ks-smoother-env` Conda environment from `environment.yml` and installs the local project in editable mode.
   ```bash
   make setup
   conda activate ks-smoother-env
   ```

## Project Structure

* `src/`: Core Python modules for the StatFEM implementation and LR-ExRTSS algorithm and execution scripts.
* `results.ipynb`: Jupyter notebook for figure generation and analysis.
* `kernel_test.ipynb`: Jupyter notebook for testing variance captures for different truncations.
* `environment.yml`: Conda environment specification.
* `.gitignore`: Configured to exclude large `.h5` and `.xdmf` data outputs.

## Usage

The experimental pipeline is governed by JSON configuration files located in `src/simulation/`. **Note:** You should modify these JSON files (e.g., to change $\kappa$ values, noise scales, or mesh resolution) before running the simulations.

### 1. Execute the Pipeline
The state estimation process follows a three-step sequence:

*   **Step A: Generate Truth (DGP)**
    Simulates the true chaotic trajectory of the 2D KSE.
    ```bash
    make ks_dgp
    ```
*   **Step B: Sample Observations**
    Samples the DGP to create the sparse, noisy observation set $\mathcal{Y}$.
    ```bash
    make ks_obs
    ```
*   **Step C: Run Prediction & Smoothing**
    Assimulates the observations into the misspecified model using the LR-ExRTSS.
    ```bash
    make ks_pred
    ```

### 2. Run All Experiments
To run the full suite of experiments defined in `src/main.py` using the current environment:
```bash
make run
```

### 3. Maintenance and Cleanup
*   **Clean build artifacts:** Remove `__pycache__` and Python build files.
    ```bash
    make clean
    ```
*   **Tear down environment:** Completely remove the Conda environment.
    ```bash
    make teardown
    ```
*   **Full Rebuild:** Useful if `environment.yml` has changed; this tears down and recreates the entire setup.
    ```bash
    make rebuild
    ```

## Data Availability

Large simulation outputs (HDF5 format) are excluded from this repository due to size constraints. The provided scripts allow for the full reconstruction of these datasets. For access to pre-computed trajectories, please contact the authors.