CC = gcc
CG = g++

#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -std=c99 -ggdb -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lgslcblas

all: iW2V iW2V_mod iW2V_mod_timer find_nearest_neighbors wordsim353_eval  

iW2V : iW2V.c
	$(CC) iW2V.c -o iW2V $(CFLAGS)

iW2V_mod : iW2V_mod.c
	$(CC) iW2V_mod.c -o iW2V_mod $(CFLAGS)

iW2V_mod_timer: iW2V_mod_timer.c
	$(CC) iW2V_mod_timer.c -o iW2V_mod_timer $(CFLAGS)

find_nearest_neighbors: Evaluation/find_nearest_neighbors.c
	$(CC) Evaluation/find_nearest_neighbors.c -o Evaluation/find_nearest_neighbors $(CFLAGS)

wordsim353_eval: Evaluation/sim-tasks/sim_eval.c
	$(CG) Evaluation/sim-tasks/sim_eval.c -o Evaluation/sim-tasks/sim_eval $(CLAGS)

test_log_prob: Evaluation/test_log_prob.c
	$(CC) Evaluation/test_log_prob.c -o Evaluation/test_log_prob $(CFLAGS)

clean:
	rm -rf iW2V iW2V_mod iW2V.dSYM *~ Evaluation/find_nearest_neighbors Evaluation/wordsim353_eval
