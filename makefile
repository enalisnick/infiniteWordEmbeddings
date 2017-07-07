CC = gcc
CG = g++

#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -std=c99 -ggdb -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lgslcblas

all: iW2V_mod iSG w2v test_iSG test_iCBOW test_SG test_CBOW 

iW2V_mod : iW2V_mod.c
	$(CC) iW2V_mod.c -o iW2V_mod $(CFLAGS)

iSG : iSG.c
	$(CC) iSG.c -o iSG $(CFLAGS)

iCBOW : iCBOW.c
	$(CC) iCBOW.c -o iCBOW $(CFLAGS)

w2v : word2vec_w_context_saving.c
	$(CC) word2vec_w_context_saving.c -o w2v $(CFLAGS)

#find_nearest_neighbors: Evaluation/find_nearest_neighbors.c
#	$(CC) Evaluation/find_nearest_neighbors.c -o Evaluation/find_nearest_neighbors $(CFLAGS)

#wordsim353_eval: Evaluation/sim-tasks/sim_eval.c
#	$(CG) Evaluation/sim-tasks/sim_eval.c -o Evaluation/sim-tasks/sim_eval $(CLAGS)

#test_log_prob: Evaluation/test_log_prob.c
#	$(CC) Evaluation/test_log_prob.c -o Evaluation/test_log_prob $(CFLAGS)

test_iSG: Perplexity/test_iSG.c 
	$(CC) Perplexity/test_iSG.c -o Perplexity/test_iSG $(CFLAGS)

test_iCBOW: Perplexity/test_iCBOW.c
	$(CC) Perplexity/test_iCBOW.c -o Perplexity/test_iCBOW $(CFLAGS)

test_SG: Perplexity/test_SG.c
	$(CC) Perplexity/test_SG.c -o Perplexity/test_SG $(CFLAGS)

test_CBOW: Perplexity/test_CBOW.c
	$(CC) Perplexity/test_CBOW.c -o Perplexity/test_CBOW $(CFLAGS)

clean:
	rm -rf iW2V_mod iW2V_mod.dSYM iSG iSG.dSYM iCBOW iCBOW.dSYM *~ Evaluation/find_nearest_neighbors Evaluation/wordsim353_eval Evaluation/W2V_test_log_prob
