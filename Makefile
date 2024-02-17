PYTHON_EXEC := env/bin/python
TORCH_DIR := env/lib/python3.11/site-packages/torch

all: $(PYTHON_EXEC) $(TORCH_DIR)
	$(MAKE) -C linear_regression

run:
	@PYTHON_EXEC=../$(PYTHON_EXEC) $(MAKE) -C linear_regression run

$(PYTHON_EXEC): 
	python -m venv env

$(TORCH_DIR): 
	env/bin/pip install torch

clean:
	$(MAKE) -C chap3 clean
	rm -rf env

.PHONY: all run clean



