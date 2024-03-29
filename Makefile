PYTHON_EXEC := env/bin/python
PYTHON_LIB := env/lib/python3.11/site-packages

TORCH_DIR := $(PYTHON_LIB)/torch
TORCHVISION_DIR := $(PYTHON_LIB)/torchvision
NUMPY_DIR := $(PYTHON_LIB)/numpy

all: $(PYTHON_EXEC) $(TORCH_DIR) $(NUMPY_DIR) $(TORCHVISION_DIR) 
	$(MAKE) -C linear_regression
	$(MAKE) -C linear_classification

run:
	@PYTHON_EXEC=../$(PYTHON_EXEC) $(MAKE) -C linear_regression run
	@PYTHON_EXEC=../$(PYTHON_EXEC) $(MAKE) -C linear_classification run

$(PYTHON_EXEC):
	python -m venv env

$(TORCH_DIR):
	env/bin/pip install torch

$(TORCHVISION_DIR):
	env/bin/pip install torchvision

$(NUMPY_DIR):
	env/bin/pip install numpy

clean:
	$(MAKE) -C linear_regression clean
	$(MAKE) -C linear_classification clean
	rm -rf env

.PHONY: all run clean
