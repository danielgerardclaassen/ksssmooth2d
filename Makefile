# Use bash for all commands
SHELL := /bin/bash

# Define the environment name (change if needed)
CONDA_ENV_NAME := ks-smoother-env

ks_dgp = src.simulation.1_ks_dgp
ks_dgp_config = src/simulation/1_ks_dgp_config.json

ks_obs = src.simulation.2_ks_obs
ks_obs_config = src/simulation/2_ks_obs_config.json

ks_pred = src.simulation.3_ks_pred
ks_pred_config = src/simulation/3_ks_pred_config.json

.PHONY: help
help:
	@echo "Makefile for the Kuramoto-Sivashinsky Smoother project"
	@echo ""
	@echo "Usage:"
	@echo "    make setup        -> Creates conda env from environment.yml and installs package"
	@echo "    make run          -> Runs the main experiment script in the conda env"
	@echo "    make clean        -> Removes python cache and build artifacts"
	@echo "    make teardown     -> Removes the conda environment and cleans artifacts"
	@echo "    make rebuild      -> Tears down the entire setup and rebuilds it from scratch"
	@echo ""

# Setup the environment from the yml file and install the local package
.PHONY: setup
setup:
	@echo "--- Creating conda environment '$(CONDA_ENV_NAME)' from environment.yml ---"
	conda env create -f environment.yml --name $(CONDA_ENV_NAME)
	@echo "\n--- Installing local project package in editable mode ---"
	conda run --no-capture-output -n $(CONDA_ENV_NAME) pip install -e .
	@echo ""
	@echo "Setup complete. Activate with: conda activate $(CONDA_ENV_NAME)"

# Run the Kuramoto-Sivashinsky DGP simulation
.PHONY: ks_dgp
ks_dgp:
	@echo "--- Running Kuramoto-Sivashinsky DGP simulation ---"
	conda run --no-capture-output -n $(CONDA_ENV_NAME) python -m $(ks_dgp) --config $(ks_dgp_config)

.PHONY: ks_obs
ks_obs:
	@echo "--- Running Kuramoto-Sivashinsky OBS simulation ---"
	conda run --no-capture-output -n $(CONDA_ENV_NAME) python -m $(ks_obs) --config $(ks_obs_config)

.PHONY: ks_pred
ks_pred:
	@echo "--- Running Kuramoto-Sivashinsky PRED simulation ---"
	conda run --no-capture-output -n $(CONDA_ENV_NAME) python -m $(ks_pred) --config $(ks_pred_config)

# Run the main script using the conda environment
.PHONY: run
run:
	@echo "--- Running main experiment script in '$(CONDA_ENV_NAME)' ---"
	conda run --no-capture-output -n $(CONDA_ENV_NAME) python src/main.py

# Clean up build artifacts
.PHONY: clean
clean:
	@echo "--- Cleaning up Python cache and build artifacts ---"
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

# Tear down the entire environment
.PHONY: teardown
teardown: clean
	@echo "--- Removing Conda environment '$(CONDA_ENV_NAME)' ---"
	-conda env remove -n $(CONDA_ENV_NAME) --yes
	@echo "Teardown complete."

# Rebuild the entire environment from scratch
.PHONY: rebuild
rebuild:
	@echo "--- Rebuilding project from scratch ---"
	$(MAKE) teardown
	$(MAKE) setup
	@echo "--- Project rebuild complete! ---"