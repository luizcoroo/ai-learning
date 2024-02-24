CFILES   = $(wildcard *.c)
CUFILES	 = $(wildcard *.cu)
OBJFILES = $(CFILES:.c=.o) $(CUFILES:.cu=.o)
OUT      = main

CC      = gcc
CFLAGS  = -O3 -Wall -I/opt/cuda/include
LDFLAGS  = -L/opt/cuda/lib
LDLIBS  = -lm -lcudart

NVCC	= /opt/cuda/bin/nvcc -ccbin=/opt/cuda/bin
CUFLAGS	= -O3 -I/opt/cuda/include

$(OUT): $(OBJFILES)

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c -o $@ $^

run: $(OUT)
	@./$(OUT)

clean:
	rm -f $(OBJFILES) $(OUT)

.PHONY: run clean
