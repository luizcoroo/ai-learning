PYTHON_EXEC:=../$(PYTHON_EXEC)

all:
	$(MAKE) -C c_cpu
	$(MAKE) -C c_cpu2

run:
	@echo "linear_classification, c, cpu, `$(MAKE) -C c_cpu run`"
	@echo "linear_classification, c, cpu2, `$(MAKE) -C c_cpu2 run`"

clean:
	$(MAKE) -C c_cpu clean
	$(MAKE) -C c_cpu2 clean

.PHONY: all run clean
