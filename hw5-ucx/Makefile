OPENMPI_VERSION := $(shell ompi_info -V 2>/dev/null | head -n1)

# Dummy Makefile
ucx:
ifdef OPENMPI_VERSION
	@echo "Found $(OPENMPI_VERSION)"
	touch ucx
else
	$(error OpenMPI not found. Please use 'module load openmpi')
endif
