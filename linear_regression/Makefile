PYTHON_EXEC:=../$(PYTHON_EXEC)

all:
	$(MAKE) -C c_cpu
	$(MAKE) -C c_cuda

run:
	@echo "linear_regression, c_cpu, `$(MAKE) -C c_cpu run`"
	@echo "linear_regression, c_cuda, `$(MAKE) -C c_cuda run`"
	@echo "linear_regression, python_cpu, `$(MAKE) -C python_cpu run`"
	@echo "linear_regression, python_gpu, `$(MAKE) -C python_gpu run`"

clean:
	$(MAKE) -C c_cpu clean
	$(MAKE) -C c_cuda clean
	$(MAKE) -C python_cpu clean
	$(MAKE) -C python_gpu clean

.PHONY: all run clean
