CFILES   = $(wildcard *.c)
OBJFILES = $(CFILES:.c=.o)
OUT      = main

CC      = gcc
CFLAGS  = -O3 -Wall
LDLIBS  = -lm

$(OUT): $(OBJFILES)

run: $(OUT)
	@./$(OUT)

clean:
	rm -f $(OBJFILES) $(OUT)

.PHONY: run clean
