CFILES   = $(wildcard src/*.c)
CUFILES	 = $(wildcard src/*.cu)
OBJFILES = $(patsubst src/%,build/%,$(CFILES:.c=.o))
OBJFILES+= $(patsubst src/%,build/%,$(CUFILES:.cu=.o))
OUT      = build/main

CC      = gcc
CFLAGS  = -O3 -Wall -isystem/opt/cuda/include -Iinclude
LDFLAGS = -L/opt/cuda/lib
LDLIBS  = -lm -lcudart

NVCC	= /opt/cuda/bin/nvcc -ccbin=/opt/cuda/bin
CUFLAGS	= -O3 -I/opt/cuda/include -Iinclude

$(OUT): $(OBJFILES)

build/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $^

build/%.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(CUFLAGS) -c -o $@ $^

run: $(OUT)
	@./$(OUT)

clean:
	rm -f $(OBJFILES) $(OUT)

.PHONY: run clean
