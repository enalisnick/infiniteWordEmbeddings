CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -ggdb -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lgslcblas

all: iW2V

iW2V : iW2V.c
	$(CC) iW2V.c -o iW2V $(CFLAGS)

clean:
	rm -rf iW2V iW2V.dSYM *~

