CONDA_ENV=job_desc_env
CONDA_CONFIG=environment.yaml
MAMBA_AVAILABLE=$(shell which mamba)

setup_env:
ifdef MAMBA_AVAILABLE
	mamba env create -f $(CONDA_CONFIG)
else
	@echo "Consider installing mamba or minimamba for better conda performance at https://github.com/mamba-org/mamba"
	conda env create -f $(CONDA_CONFIG)
endif


update_env: export_env
ifdef MAMBA_AVAILABLE
	mamba env update $(CONDA_ENV) -f $(CONDA_CONFIG) $(OPT)
else
	@echo "Consider installing mamba or minimamba for better conda performance at https://github.com/mamba-org/mamba"
	conda env update $(CONDA_ENV) -f $(CONDA_CONFIG)
endif

export_env:
ifdef MAMBA_AVAILABLE
	mamba env export --no-builds | grep -v "prefix"  > $(CONDA_CONFIG)
else
	@echo "Consider installing mamba or minimamba for better conda performance at https://github.com/mamba-org/mamba"
	conda env export --no-builds | grep -v "prefix" | sed > $(CONDA_CONFIG)
endif