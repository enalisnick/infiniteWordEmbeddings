CC = gcc
CG = g++

#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -std=c99 -ggdb -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lgslcblas

all: iW2V_mod iSG find_nearest_neighbors wordsim353_eval

iW2V_mod : iW2V_mod.c
	$(CC) iW2V_mod.c -o iW2V_mod $(CFLAGS)

iSG : iSG.c
	$(CC) iSG.c -o iSG $(CFLAGS)

iCBOW : iCBOW.c
	$(CC) iCBOW.c -o iCBOW $(CFLAGS)

find_nearest_neighbors: Evaluation/find_nearest_neighbors.c
	$(CC) Evaluation/find_nearest_neighbors.c -o Evaluation/find_nearest_neighbors $(CFLAGS)

wordsim353_eval: Evaluation/sim-tasks/sim_eval.c
	$(CG) Evaluation/sim-tasks/sim_eval.c -o Evaluation/sim-tasks/sim_eval $(CLAGS)

test_log_prob: Evaluation/test_log_prob.c
	$(CC) Evaluation/test_log_prob.c -o Evaluation/test_log_prob $(CFLAGS)

W2V_test_log_prob: Evaluation/W2V_test_log_prob.c
	$(CC) Evaluation/W2V_test_log_prob.c -o Evaluation/W2V_test_log_prob $(CFLAGS)

clean:
	rm -rf iW2V_mod iW2V_mod.dSYM iSG iSG.dSYM iCBOW iCBOW.dSYM *~ Evaluation/find_nearest_neighbors Evaluation/wordsim353_eval Evaluation/W2V_test_log_prob
