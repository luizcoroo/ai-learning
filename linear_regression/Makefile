PYTHON_EXEC:=../$(PYTHON_EXEC)

all:
	$(MAKE) -C c_cpu
	$(MAKE) -C c_gpu

run:
	@echo "linear_regression, c, cpu, `$(MAKE) -C c_cpu run`"
	@echo "linear_regression, c, gpu, `$(MAKE) -C c_gpu run`"
	@echo "linear_regression, python, cpu, `$(MAKE) -C python_cpu run`"
	@echo "linear_regression, python, gpu, `$(MAKE) -C python_gpu run`"

clean:
	$(MAKE) -C c_cpu clean
	$(MAKE) -C c_gpu clean
	$(MAKE) -C python_cpu clean
	$(MAKE) -C python_gpu clean

.PHONY: all run clean
