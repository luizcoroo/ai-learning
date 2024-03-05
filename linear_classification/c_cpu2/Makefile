CFILES   = $(wildcard src/*.c)
OBJFILES = $(patsubst src/%,build/%,$(CFILES:.c=.o))
OUT      = build/main

CC      = gcc
CFLAGS  = -std=c2x -O3 -Wall -Iinclude
LDLIBS  = -lm 

$(OUT): $(OBJFILES)

build/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $^ -o $@

run: $(OUT)
	@./$(OUT)

clean:
	rm -f $(OBJFILES) $(OUT)

.PHONY: run clean
