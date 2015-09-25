CC = gcc
CG = g++

#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -std=c99 -ggdb -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lgslcblas

all: iW2V find_nearest_neighbors wordsim353_eval

iW2V : iW2V.c
	$(CC) iW2V.c -o iW2V $(CFLAGS)

find_nearest_neighbors: Evaluation/find_nearest_neighbors.c
	$(CC) Evaluation/find_nearest_neighbors.c -o Evaluation/find_nearest_neighbors $(CFLAGS)

wordsim353_eval: Evaluation/wordsim353_eval.c
	$(CG) Evaluation/wordsim353_eval.c -o Evaluation/wordsim353_eval 

clean:
	rm -rf iW2V iW2V.dSYM *~ Evaluation/find_nearest_neighbors Evaluation/wordsim353_eval
