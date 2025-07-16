CONDA_ENV=soft-skills-env
CONDA_CONFIG=environment.yml
MAMBA_AVAILABLE=$(shell which mamba)
CONDA_AVAILABLE=$(shell which conda)
SHELL=/bin/zsh

.PHONY: setup_env update_env export_env clean_env clean_kernel list_env activate_env check_conda install_kernel help

# Create a new environment
setup_env: check_conda
ifdef MAMBA_AVAILABLE
	mamba env create -f $(CONDA_CONFIG)
else
	@echo "Consider installing mamba for better performance: https://github.com/mamba-org/mamba"
	conda env create -f $(CONDA_CONFIG)
endif
	$(MAKE) install_kernel

# Update an existing environment
update_env: check_conda export_env
ifdef MAMBA_AVAILABLE
	mamba env update -n $(CONDA_ENV) -f $(CONDA_CONFIG)
else
	@echo "Consider installing mamba for better performance: https://github.com/mamba-org/mamba"
	conda env update -n $(CONDA_ENV) -f $(CONDA_CONFIG)
endif
	$(MAKE) install_kernel

# Export the current environment
export_env: check_conda
ifdef MAMBA_AVAILABLE
	mamba env export --no-builds | grep -v "prefix" > $(CONDA_CONFIG)
else
	conda env export --no-builds | grep -v "prefix" > $(CONDA_CONFIG)
endif

# Install Jupyter kernel
install_kernel:
ifndef CONDA_PREFIX
	$(error "CONDA_PREFIX is not set. Ensure the environment $(CONDA_ENV) is activated.")
endif
	$(CONDA_PREFIX)/bin/python -m ipykernel install --user --name $(CONDA_ENV) --display-name "Python ($(CONDA_ENV))"

# Remove the environment
clean_env: check_conda
	conda env remove -n $(CONDA_ENV) || echo "Environment $(CONDA_ENV) does not exist."

# Remove Jupyter kernel
clean_kernel:
	jupyter kernelspec remove $(CONDA_ENV) -f || echo "Kernel $(CONDA_ENV) does not exist."

# List all environments
list_env: check_conda
	conda env list

# Show activation instructions
activate_env:
ifdef MAMBA_AVAILABLE
	@echo "To activate this environment, use: mamba activate $(CONDA_ENV)"
else
	@echo "To activate this environment, use: conda activate $(CONDA_ENV)"
endif

# Check if conda is available
check_conda:
ifndef CONDA_AVAILABLE
	$(error "conda not found. Please install conda first.")
endif

# Help target
help:
	@echo "Available targets:"
	@echo "  setup_env    : Create a new conda environment from $(CONDA_CONFIG)"
	@echo "  update_env   : Update existing conda environment from $(CONDA_CONFIG)"
	@echo "  export_env   : Export current conda environment to $(CONDA_CONFIG)"
	@echo "  install_kernel : Install Jupyter kernel for the environment"
	@echo "  clean_env    : Remove conda environment"
	@echo "  clean_kernel : Remove Jupyter kernel"
	@echo "  list_env     : List all conda environments"
	@echo "  activate_env : Show command to activate environment"
	@echo "  check_conda  : Verify if conda is installed"
	@echo "  help         : Show this help message"

# Default target
.DEFAULT_GOAL := help

