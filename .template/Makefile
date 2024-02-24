PYTHON_EXEC:=../$(PYTHON_EXEC)

all:
	$(MAKE) -C c_cpu
	$(MAKE) -C c_gpu

run:
	@echo "chapter_name, c, cpu, `$(MAKE) -C c_cpu run`"
	@echo "chapter_name, c, gpu `$(MAKE) -C c_gpu run`"
	@echo "chapter_name, python, cpu, `$(MAKE) -C python_cpu run`"
	@echo "chapter_name, python, gpu, `$(MAKE) -C python_gpu run`"

clean:
	$(MAKE) -C c_cpu clean
	$(MAKE) -C c_gpu clean
	$(MAKE) -C python_cpu clean
	$(MAKE) -C python_gpu clean

.PHONY: all run clean
