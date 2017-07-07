#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "eval_lib.h"

const int DEFAULT_K = 50;

int main(int argc, char **argv) {
  char file_name[max_size];
  long long vocab_size, embed_size;
  char *vocab;
  float *vectors;
  int k = DEFAULT_K;
 
  if (argc < 2) {
    printf("Input file not found\n");
    return -1;
  }
  if (argc > 2) k = atoi(argv[2]);
  
  strcpy(file_name, argv[1]);
  read_vectors(file_name, &vocab_size, &embed_size, &vocab, &vectors);  // read vocab & vectors from file
  
  char str[max_w];
  int a = 0;
  char *nn[k];
  for (int i = 0; i < k; i++) nn[i] = (char *)malloc(max_w * sizeof(char));
  float bestd[k];
  int dim_used[k];
  while (1) {
    printf("Enter word (EXIT to break): ");
    a = 0;
    while (1) {
      str[a] = fgetc(stdin);
      if ((str[a] == '\n') || (a >= max_size - 1)) {
        str[a] = 0;
        break;
      }
      a++;
    }
    
    // Get word to search
    if (!strcmp(str, "EXIT")) break;
    int b = find_str(vocab, vocab_size, str);
    if (b == -1) {
      printf("Out of dictionary word!\n");
      continue;
    }
    printf("\nWord: %s  Position in vocabulary: %d\n", str, b); 
    
    // Find nearest-neighbors of word 
    get_k_sim(vocab, vectors, vocab_size, embed_size, b, k, nn, bestd, dim_used);
    print_nn(nn, bestd, dim_used, k);
    printf("#Non-zero Dims: %d\n", num_non_zero_dims(vectors, embed_size, b));  
  }
  return 0;
}

