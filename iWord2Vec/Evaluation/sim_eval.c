#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "eval_lib.h"

int my_pair_cmp(const void *const first, const void *const second);

const char* WORD_SIM353_FILE = "Evaluation/wordSim353_sorted.csv";
const int WORD_SIM_LEN = 353;

const char *MEN_FILE = "Evaluation/MEN_sorted.txt";
const int MEN_LEN = 3000;

struct pair { 
  double sim;
  int index;
};
 
int main(int argc, char **argv) {  
  char file_name[max_size];
  long long vocab_size, embed_size;
  char *vocab;
  double *vectors;
  int is_MEN = 0;
  int full_dim = 0;

  if (argc < 2) {
    printf("Input file not found\n");
    return -1;
  }
  if (argc > 2) {
    full_dim = atoi(argv[2]);
  }
  if (argc > 3) {
    is_MEN = atoi(argv[3]);
  }  
  const char *sim_file_name;
  int len;
  if (is_MEN == 0) {
    sim_file_name = WORD_SIM353_FILE;
    len = WORD_SIM_LEN;
  }
  else {
    sim_file_name = MEN_FILE;
    len = MEN_LEN;
  } 
  strcpy(file_name, argv[1]);
  read_vectors(file_name, &vocab_size, &embed_size, &vocab, &vectors);  // read vocab & vectors from file
 
  printf("Using word vectors from %s\n", file_name);
  if (full_dim == 0) {
    printf("Using prob derived dims\n");
  } else {
    printf("Using all dims\n");
  } 
  printf("Using sim benchmark %s\n", sim_file_name);
  // Read sim words from file
  FILE *f;
  f = fopen(sim_file_name, "rb"); 
  struct pair w2v_sim[len];
  double human_sim[len];
  int valid = 0;
  for (int i = 0; i < len; i++) { 
    char word1[max_size] = {0};
    char word2[max_size] = {0};
    double sim = 0.0;
  
    read_str(word1, f);
    read_str(word2, f);
    
    fscanf(f, "%lf", &sim);
    human_sim[i] = sim;
    fgetc(f);
    
    if (is_MEN == 0)  fgetc(f);  // MEN requires one less fgetc

    int idx1 = find_str(vocab, vocab_size, word1);
    int idx2 = find_str(vocab, vocab_size, word2);
    // Keep track of valid examples where both words in vocab 
    if (idx1 != -1 && idx2 != -1) {
      if (full_dim == 0) {
        int mode_z = get_mode_z(vectors, embed_size, idx1, idx2);
        w2v_sim[i].sim = dot_product_sim(vectors, embed_size, idx1, idx2, mode_z); 
      } else {  
        w2v_sim[i].sim = cosine_sim(vectors, embed_size, idx1, idx2); 
      }
      w2v_sim[i].index = valid;
      valid++;
    }
    else {
      w2v_sim[i].sim = -1;
    }
  }
  
  // Copy valid sims
  struct pair final_w2v_sim[valid]; 
  int cnt = 0;
  for (int i = 0; i < len; i++) {
    if (w2v_sim[i].sim != -1) {
      final_w2v_sim[cnt].sim = w2v_sim[i].sim;
      final_w2v_sim[cnt].index = w2v_sim[i].index;
      cnt++;
    }
  }

  // Sort w2v sims & calculate rank correlation
  qsort((void *) final_w2v_sim, valid, sizeof(*final_w2v_sim), my_pair_cmp);
  double sum_squ_dist = 0.0;
  for (int i = 0; i < valid; i++) {
    sum_squ_dist += (i - final_w2v_sim[i].index) * (i - final_w2v_sim[i].index);
  }
  printf("%lf\n", 1.0 - (6*sum_squ_dist)/(valid*(valid*valid - 1.0)));
  
  return 0;
}

int my_pair_cmp(const void *const first, const void *const second)
{
  const pair* a = (const pair*)first;
  const pair* b = (const pair*)second;
  if (a->sim > b->sim)
    return 1;
  else if (a->sim < b->sim)
    return -1;
  else
    return 0;
}
