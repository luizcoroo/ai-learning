PYTHON_EXEC:=../$(PYTHON_EXEC)

all:
	$(MAKE) -C c_cpu
	$(MAKE) -C c_cuda

run:
	@echo "linear_regression, c_cpu, `$(MAKE) -C c_cpu run`"
	@echo "linear_regression, c_cuda, `$(MAKE) -C c_cuda run`"
	@echo "linear_regression, python, `$(MAKE) -C python run`"

clean:
	$(MAKE) -C c_cpu clean
	$(MAKE) -C c_cuda clean
	$(MAKE) -C python clean

.PHONY: all run clean
