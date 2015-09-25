#include <math.h>

const long long max_size = 2000;  // max length of strings
const long long max_w = 50;  // max length of vocabulary entries

void read_str(char *word, FILE *f) {
  int i = 0;
  while(1) {
    char c = fgetc(f);
    if (c != ' ' && c != ',' && c != '\n') { word[i] = c; }
    else {
      break;
    }
    i++;
  }
}

int find_str(char *vocab, long long vocab_size, char *str) {
  int b;
  for (b = 0; b < vocab_size; b++) if (!strcmp(&vocab[b * max_w], str)) break;
  if (b == vocab_size) b = -1;
  return b;
}

void print_nn(char **nn, double *bestd, int k) {
  printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
  for (int i = 0; i < k; i++) printf("%50s\t\t%lf\n", nn[i], bestd[i]);
}

int num_non_zero_dims(double *vectors, long long embed_size, int idx) {
  int num = 0;
  for (int i = 0; i < embed_size; i++) {
    if (vectors[idx * embed_size + i] != 0)  num++;
  }
  return num;
}

double cosine_sim(double *vectors, long long embed_size, int idx1, int idx2) {
  double prod = 0.0;
  for (int i = 0; i < embed_size; i++) {
    prod += vectors[idx1 * embed_size + i] * vectors[idx2 * embed_size + i];
  }
  return prod;
}

void get_k_sim(char *vocab, double *vectors, long long vocab_size, long long embed_size, int word_idx, int k, char **nn, double *bestd) {
  for (int i = 0; i < k; i++) {
    bestd[i] = -1;
    nn[i][0] = -1;
  }
  for (int i = 0; i < vocab_size; i++) {
    if (i != word_idx) {
      double dist = cosine_sim(vectors, embed_size, word_idx, i);
      for (int j = 0; j < k; j++) {
        if (dist > bestd[j]) {
          // copy vocab & distances down
          for (int l = k-1; l > j; l--) {
            bestd[l] = bestd[l-1];
            strcpy(nn[l], nn[l-1]);
          }
          // replace at spot
          strcpy(nn[j], &vocab[i * max_w]); 
          bestd[j] = dist;
          break;
        }
      }  
    }
  }
}

void read_vectors(char *file_name, long long *vocab_size, long long *embed_size, char **vocab, double **vectors) {
  FILE *f;
  f = fopen(file_name, "rb");
  // Read Header info 
  fscanf(f, "%lld", vocab_size);
  fscanf(f, "%lld", embed_size);

  *vocab = (char *)malloc(*vocab_size * max_w * sizeof(char));
  *vectors = (double *)malloc(*vocab_size * *embed_size * sizeof(double));
  for (int i = 0; i < *vocab_size; i++) {
    fgetc(f); // read '\n'
    int j = 0;
    while (1) { // read word
      char c = fgetc(f);
      if (c != ' ' && c != '\n') { (*vocab)[i * max_w + j] = c; } 
      else {
        break;
      }
      j++;
    }
    //printf("%s\n", *vocab + i * max_w);
    // store unit vector for cosine similarity
    double len = 0;
    for (int j = 0; j < *embed_size; j++) { // read vector
      fscanf(f, "%lf", &(*vectors)[i * *embed_size + j]);
      len += (*vectors)[i * *embed_size + j] * (*vectors)[i * *embed_size + j]; 
    }
    len = sqrt(len);
    for (int j = 0; j < *embed_size; j++) {
      (*vectors)[i * *embed_size + j] /= len;
    }
    fgetc(f); // read space
  }
}

void print_vector(char *vocab, double *vectors, long long embed_size, int idx) {
  char str[max_size];
  strcpy(str, &vocab[idx * max_w]);
  printf("%s\n", str);
  printf("[");
  for (int i = 0; i < embed_size; i++) {
    printf("%f ", vectors[idx * embed_size + i]);
  }
  printf("]\n");
}
