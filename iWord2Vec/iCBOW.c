#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_randist.h>
//#include "Evaluation/eval_lib.h"

// Global Variables
#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  char *word;
};

// pthread only allows passing of one argument
typedef struct {
  int id;
} ThreadArg;
ThreadArg *thread_arg;

char train_file[MAX_STRING], output_file[MAX_STRING], context_output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, window = 5, min_count = 1, num_threads = 1, min_reduce = 1;
real dim_penalty = 1.1;
float global_train_loss = 0.0;
float log_dim_penalty; //we'll compute this in the training function
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, embed_max_size = 750, embed_current_size = 5, global_loss_diff = 0;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, *alpha_count_adjustment;
real alpha = 0.05, starting_alpha, sample = 1e-3, sparsity_weight = 0.001;
real *input_embed, *context_embed, *alpha_per_dim;
clock_t start;
int negative = 5;
int num_z_samples = 5;

const int table_size = 1e8;
const double epsilon = 1e-8;
int *table;

const int EXP_LEN = 200;
float *exp_table; 

// adadelta variables
int adadelta_flag = 0; // 1 if we want to use AdaDelta, 0 for regular SGD
float *input_grad_history, *context_grad_history;
float rho_adadelta = 0.95; // taken from adadelta paper
float epsilon_adadelta = 0.1; // taken from https://github.com/jeevanshankar1991/Word2Vec

/*
  Build table which precompute exp function for certain integer
  values
*/
void build_exp_table() {
  exp_table = (float *) calloc(EXP_LEN * 2 + 1, sizeof(float));
  for (int i = -EXP_LEN; i <= EXP_LEN; i++) {
    exp_table[i + EXP_LEN] = exp(i);
  }
}

/*
  Exp approximate function from 
  http://stackoverflow.com/questions/10552280/fast-exp-calculation-possible-to-improve-accuracy-without-losing-too-much-perfo/14143184#14143184
  Error in input range [-1,1] is 0.36%
*/
float exp_approx(float x) { 
  return (24+x*(24+x*(12+x*(4+x))))*0.041666666f;
}

/*
  Separate into integer and decimal components and use table and 
  approximate exp function to compute each part, respectively
*/
float exp_fast(float x) {
  int x_int = (int)x;
  float x_dec = x - x_int;

  float exp_table_val = 0.0;
  if (x_int < -EXP_LEN) {
    exp_table_val = exp_table[0];
  } 
  else if (x_int > EXP_LEN) {
    exp_table_val = exp_table[2*EXP_LEN];
  }
  else {
    exp_table_val = exp_table[x_int + EXP_LEN];
  }
 
  return exp_table_val * exp_approx(x_dec);   
}

// Build table from which to rand. sample words
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  // initialize context embeddings
  a = posix_memalign((void **)&context_embed, 128, (long long)vocab_size * embed_max_size * sizeof(real));
  if (context_embed == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < vocab_size; a++) for (b = 0; b < embed_max_size; b++) {
      // random (instead of zero) to avoid multi-threaded problems
      next_random = next_random * (unsigned long long)25214903917 + 11;
      context_embed[a * embed_max_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / embed_current_size; 
  }
  // initialize input embeddings
  a = posix_memalign((void **)&input_embed, 128, (long long)vocab_size * embed_max_size * sizeof(real));
  for (a = 0; a < vocab_size; a++) for (b = 0; b < embed_max_size; b++) {
      // only initialize first few dims so we can tell the true vector length
      if (b < embed_current_size){
	next_random = next_random * (unsigned long long)25214903917 + 11;
	input_embed[a * embed_max_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / embed_current_size;
      }
      else{
	input_embed[a * embed_max_size + b] = 0.0;
      }
  }
  
  // intialize gradient history vectors if using adadelta 
  if (adadelta_flag == 1){
    input_grad_history = calloc(vocab_size * embed_max_size, sizeof(float)); 
    context_grad_history = calloc(vocab_size * embed_max_size, sizeof(float));
  }

  // initialize per dimension learning rate array
  alpha_per_dim = (real *) calloc(embed_max_size, sizeof(real));
  for (b = 0; b < embed_max_size; b++) alpha_per_dim[b] = alpha;
  alpha_count_adjustment = (long long *) calloc(embed_max_size, sizeof(long long));
}

// function to compute E(w, c, z)
float compute_energy(long long w_idx, long long c_idx, int z){
  long long a;
  float energy = 0.0;
  for (a = 0; a<z; a++) energy += 
    -input_embed[w_idx + a]*context_embed[c_idx + a] + log_dim_penalty 
    + sparsity_weight*input_embed[w_idx + a]*input_embed[w_idx+a] 
    + sparsity_weight*context_embed[c_idx + a]*context_embed[c_idx+a];
  return energy;
}

/*
  Compute e^(-E(w,c,z)) for z=1,...,curr_z,curr_z+1  
  -> dist: float array to fill; should be of size curr_z+1 
  -> w_idx: word index
  -> c_idx: context index
  -> curr_z: current number of dimensions 
*/
// (unnormProbs_z_given_w_C, pos_context_store, a, pos_context_counter, local_embed_size_plus_one - 1)
float compute_z_dist(float *dist, long long *context, int center_idx, int context_size, int curr_z) { 
  float norm = 0.0;
  long long w_idx = context[center_idx] * embed_max_size;
  for (int a = 0; a < curr_z; a++) {
    // precompute context values
    float context_sum = 0;
    float context_norms = 0;
    for (int j = 0; j < context_size; j++){
      if (j == center_idx) continue;
      long long c_idx = context[j] * embed_max_size;
      context_sum += context_embed[c_idx + a];
      context_norms += context_embed[c_idx + a] * context_embed[c_idx + a];
    }
    // compute entergy
    float val = -input_embed[w_idx + a]*context_sum 
      +(context_size-1)*log_dim_penalty + (context_size-1)*sparsity_weight*input_embed[w_idx + a]*input_embed[w_idx + a] 
                +sparsity_weight*context_norms;
    for (int b = a; b <= curr_z; b++) {
      dist[b] += val;
    }
    dist[a] = exp_fast((1.0/(context_size - 1.0)) * -dist[a]);
    norm += dist[a];
  }
  dist[curr_z] = (dim_penalty / (dim_penalty - 1.0)) * exp_fast((1.0/(context_size - 1.0)) * -dist[curr_z]);
  norm += dist[curr_z];

  return norm;
}

void compute_p_z_given_w_C(float *prob_z_given_w_C, float *sum_prob_z_given_w_C, long long *context, int center_idx, int context_size, int curr_z) {
  float norm = compute_z_dist(prob_z_given_w_C, context, center_idx, context_size, curr_z);
  float sum = 0.0;
  for (int a = curr_z; a >= 0; a--) {
    prob_z_given_w_C[a] = prob_z_given_w_C[a]/norm;
    sum += prob_z_given_w_C[a];
    sum_prob_z_given_w_C[a] = sum;
  }
}

// prob_c_z_given_w should be of size true_context_size * curr_z_plus_one
//(a, pos_context_store, negative_list, pos_context_counter, negative, prob_w_z_given_C, local_embed_size_plus_one)
void compute_p_w_z_given_C(long long center_idx, long long *context, long long *negatives, int context_size, int negative_size, float *prob_w_z_given_C, float *sum_prob_w_z_given_C, int curr_z_plus_one) {
  // compute e^(-E(w,c,z)) for z = 1,...,curr_z,curr_z+1 for every context c
  long long save_true_w = context[center_idx];
  float norm = 0.0;
  norm += compute_z_dist(prob_w_z_given_C, context, center_idx, context_size, curr_z_plus_one - 1);
  for (int s = 0; s < negative_size; s++) {
    context[center_idx] = negatives[s];
    norm += compute_z_dist(prob_w_z_given_C + (s+1) * curr_z_plus_one, context, center_idx, context_size, curr_z_plus_one - 1); 
  }
  // replace true center word
  context[center_idx] = save_true_w;
  // z_dist_list should now have the prob. of each dim for every context word 
  
  // compute prob
  for (int s = 0; s < negative_size + 1; s++) {
    float sum = 0.0;
    for (int z = curr_z_plus_one-1; z >= 0; z--) {
      prob_w_z_given_C[s * curr_z_plus_one + z] = prob_w_z_given_C[s * curr_z_plus_one + z]/norm;
      sum += prob_w_z_given_C[s * curr_z_plus_one + z];
      sum_prob_w_z_given_C[s * curr_z_plus_one + z] = sum;
    }
  }
}

// function to sample value of z_hat -- modified but essentially coppied from StackOverflow 
// http://stackoverflow.com/questions/25363450/generating-a-multinomial-distribution
int sample_from_mult(double probs[], int k, const gsl_rng* r){ // always sample 1 value
  unsigned int mult_op[k];
  gsl_ran_multinomial(r, k, 1, probs, mult_op);
  for (int idx=1; idx<=k; idx++){
    if (mult_op[idx-1]==1) return idx;
  }
  return 0;
}

// Samples N values of z_hat and returns in vals list and returns the max of samples 
// Params:
// probs -- unnormalized probabilities
// k -- l+1 (size of current embedding + 1)
// N -- number of samples to draw
// vals -- N-size array containing the sampled values
// r -- random number generator 
int sample_from_mult_list(float float_probs[], int k, int vals[], int N, 
  const gsl_rng* r) {
  unsigned int mult_op[k];
  // copy over to double array
  double probs[k];
  for (int i = 0; i < k; i++) {
    probs[i] = float_probs[i];
  }

  gsl_ran_multinomial(r, k, N, probs, mult_op); // returns array of size N with amount of each value sampled 
  int cnt = 0;
  // add sampled values to list 
  int max_idx = -1;
  for (int idx=1; idx<=k; idx++) {  
    if (mult_op[idx-1] > 0) { // get max so far
      max_idx = idx;
    }
    // We know amount of this value, so add this many to list
    for (int num=0; num<mult_op[idx-1]; num++) {
      vals[cnt] = idx;
      cnt++;
    }
  }
  return max_idx;
}

void check_value(float val, char *name, int idx) {
  if (isnan(val) || isinf(val)) { 
    printf("-------------------------\n");
    if (isnan(val)) printf("NAN!\n");
    else if (isinf(val)) printf("INF!\n");
 
    printf("idx: %d, name=%s, val=%f\n", idx, name, val);
    printf("-------------------------\n");
    fflush(stdout);
    exit(1);
  }
}

void reset(char *DEBUG, FILE *debug_file) {
  fclose(debug_file);
  debug_file = fopen(DEBUG, "w+"); 
  fseek(debug_file, 0, SEEK_SET);
}

void write_str(FILE *debug_file, char *name) {
  fprintf(debug_file, "-------------------------------------------\n");
  fprintf(debug_file, "%s\n", name);
  fprintf(debug_file, "-------------------------------------------\n");
}

void write_float(FILE *debug_file, char *name, float val, int idx) {
  fprintf(debug_file, "%s (%d): %f\n", name, idx, val);
}

void write_arr(FILE *debug_file, char *name, float *arr, int num_dims, ...) {
  va_list valist;
  va_start(valist, num_dims);

  fprintf(debug_file, "-------------------------------------------\n");
  fprintf(debug_file, "NAME: %s\n", name);
  if (num_dims == 1) {
    int len = va_arg(valist, int);
    for (int i = 0; i < len; i++) {
      fprintf(debug_file, "%d: %f\n", i, arr[i]);
    } 
  }
  else if (num_dims == 2) {
    int len1 = va_arg(valist, int);
    int len2 = va_arg(valist, int);
    for (int i = 0; i < len1; i++) {
      fprintf(debug_file, "%d: \n", i);
      for (int j = 0; j < len2; j++) {
        fprintf(debug_file, "\t%d: %f\n", j, arr[i * len2 + j]);
      }
    }
  } 
  fprintf(debug_file, "*******************************************\n");
}

void print_args() {
  printf("# TRAINING SETTINGS #\n"); 
  printf("Train Corpus: %s\n", train_file);
  printf("Output file: %s\n", output_file);
  printf("Num. of threads: %d\n", num_threads);
  printf("Initial dimensionality: %lld\n", embed_current_size);
  printf("Max dimensionality: %lld\n", embed_max_size); 
  printf("Context window size: %d\n", window); 
  printf("Num. of negative samples: %d\n", negative); 
  printf("Training iterations (epochs): %lld\n", iter); 
  printf("Learning rate: %f\n", (float)alpha ); 
  printf("Dimension penalty: %f\n", (float)dim_penalty); 
  printf("Sparsity weight: %f\n", (float)sparsity_weight);
  printf("#####################\n");
  fflush(stdout);
}

void save_vectors(char *output_file, long long int vocab_size, long long int embed_current_size, struct vocab_word *vocab, real *input_embed) {
  FILE *fo;
  fo = fopen(output_file, "wb");
  // Save the word vectors
  fprintf(fo, "%lld %lld\n", vocab_size, embed_current_size);
  long a,b;
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    // only print the non-zero dimensions
    for (b = 0; b < embed_current_size; b++) fprintf(fo, "%f ", input_embed[a * embed_max_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void *TrainModelThread(void *arg) {
  // get thread arguments
  ThreadArg *thread_arg = (ThreadArg *)arg;
  int id = thread_arg->id;
  printf("%d\n", thread_arg->id);
  // Debug file
  char DEBUG[12];
  sprintf(DEBUG, "debug_%d.txt", id);
  FILE *debug_file;
  const int RESET_CNT = 501;
  debug_file = fopen(DEBUG, "w+");

  long long a, b, d, word, center_word, last_word, negative_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1], pos_context_counter;
  long long center_word_position, context_word_position, neg_center_word_position, z_max, c, local_iter = iter;
  float log_prob_per_word = 0;
  unsigned long long next_random = (long long)id;
  clock_t now;

  // open corpus file and seek to thread's position in it
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  
  // set up random number generator
  const gsl_rng_type * T2;
  gsl_rng * r2;
  srand(time(NULL));
  unsigned int Seed2 = rand();
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r2 = gsl_rng_alloc (T2);
  gsl_rng_set (r2, Seed2);

  int *z_samples = (int *) calloc(num_z_samples, sizeof(int)); // M-sized array of sampled z values
  long long *negative_list = (long long *) calloc(negative, sizeof(long long));
  long long *pos_context_store = (long long *) calloc(2*window+1, sizeof(long long));
  // terms needed for p(z|w,C)
  float *probs_z_given_w_C = (float *) calloc(embed_max_size, sizeof(float));
  float *sum_probs_z_given_w_C = (float *) calloc(embed_max_size, sizeof(float));

  float train_log_probability = 0.0;  // track if model is learning 
  while (1) {
    // track training progress
    if (word_count - last_word_count > 10000) { // TODO: lowered for debugging
      long long diff = word_count - last_word_count;
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha_per_dim[embed_current_size-1],
	       word_count_actual / (real)(iter * train_words + 1) * 100,
	       word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
	global_loss_diff += diff;
	global_train_loss += train_log_probability;
        printf("loss: %f  ", global_train_loss / global_loss_diff);
	printf("curr dim: %lld\n", embed_current_size);	
        fflush(stdout);
        train_log_probability = 0.0;
      }
      if (adadelta_flag != 1){
	for (c = 0; c < embed_current_size; c++){
	  alpha_per_dim[c] = starting_alpha * (1 - (word_count_actual - alpha_count_adjustment[c]) / (real)(iter * train_words - alpha_count_adjustment[c] + 1));
	  if (alpha_per_dim[c] < starting_alpha * 0.0001) alpha_per_dim[c] = starting_alpha * 0.0001;
	}
      }
    }

    // reset debug file
    if (word_count_actual % RESET_CNT == 0) { reset(DEBUG, debug_file); }

    // read a new sentence / line
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking the same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    // if EOF, reset to beginning
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    
    // get ALL context words (including word to be predicted (w))
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window; // Samples(!) window size 
    pos_context_counter = 0;
    for (a = b; a < window * 2 + 1; a++){
	c = sentence_position - window + a;
	if (c < 0) continue;
	if (c >= sentence_length) continue;
	last_word = sen[c];
	if (last_word <= 0) continue;
	pos_context_store[pos_context_counter] = last_word;
	pos_context_counter++; 
    }

    // Check that we found some positive context words
    // If not, get a new sentence (ie continue)
    if (pos_context_counter < 3) {
      sentence_position++;
      if (sentence_position >= sentence_length) {
	sentence_length = 0;
      }
      continue;
    }

    float window_normalization = 1.0/(pos_context_counter-1.0);
 
    // MAIN LOOP THROUGH POSITIVE CONTEXT
    log_prob_per_word = 0.0;
    for (a = 0; a < pos_context_counter; a++) { 

      // center word to predict
      center_word = pos_context_store[a];
      center_word_position = center_word * embed_max_size;

      // lock-in value of embed_current_size for thread since its shared globally                                                    
      int local_embed_size_plus_one = embed_current_size + 1;
      
      // initialize local vars needed for p(w,z|c1,...,cK) and grad of w,c1,...,cK
      float *prob_w_z_given_C = (float *) calloc(local_embed_size_plus_one * (negative + 1), sizeof(float));
      float *sum_prob_w_z_given_C = (float *) calloc(local_embed_size_plus_one * (negative + 1), sizeof(float));
      float *gradient = (float *) calloc(local_embed_size_plus_one * pos_context_counter, sizeof(float));
      float *neg_gradient = (float *) calloc(local_embed_size_plus_one * negative, sizeof(float));     
 
      // only need to initialize dimensions less than current_size + 1 since that's all it can grow                                                          
      // we'd like to do this after the last gradient update but local_embed_size_plus_one may have grew, leaving old values 
      for (c = 0; c < local_embed_size_plus_one; c++) {
        probs_z_given_w_C[c] = 0.0;
        sum_probs_z_given_w_C[c] = 0.0;
      }
      // compute p(z|w,c1,..cK)
      compute_p_z_given_w_C(probs_z_given_w_C, sum_probs_z_given_w_C, pos_context_store, a, pos_context_counter, local_embed_size_plus_one - 1); 
      write_arr(debug_file, "p(z|w,C)", probs_z_given_w_C, 1, local_embed_size_plus_one);
      write_arr(debug_file, "sum p(z|w,C)", sum_probs_z_given_w_C, 1, local_embed_size_plus_one);

      // sample z: z_hat ~ p(z|w,c1,...,cK) and expand if necessary
      // no need to normalize, function does it for us
      z_max = sample_from_mult_list(probs_z_given_w_C, 
                  local_embed_size_plus_one, z_samples, num_z_samples, r2);
      if (z_max == local_embed_size_plus_one 
              && embed_current_size < local_embed_size_plus_one 
              && z_max < embed_max_size) {
	alpha_count_adjustment[embed_current_size] = word_count_actual;
	embed_current_size++;
      }

      // NEGATIVE SAMPLING CENTER WORDS
      d = negative-1;
      while (d >= 0) {
	negative_list[d] = 0; // clear old contexts
	next_random = next_random * (unsigned long long)25214903917 + 11;
	negative_word = table[(next_random >> 16) % table_size];
	if (negative_word == 0) negative_word = next_random % (vocab_size - 1) + 1;
	if (negative_word == center_word || negative_word <= 0) continue; 
	negative_list[d] = negative_word;
        d--;
      }

      // compute p(w,z|c1,...,cK)
      compute_p_w_z_given_C(a, pos_context_store, negative_list, pos_context_counter, negative, prob_w_z_given_C, 
        sum_prob_w_z_given_C, local_embed_size_plus_one);
      write_arr(debug_file, "p(w,z|C)", prob_w_z_given_C, 2, negative + 1, local_embed_size_plus_one);
      write_arr(debug_file, "sum p(w,z|C)", sum_prob_w_z_given_C, 2, negative + 1, local_embed_size_plus_one);
 
      // compute p(w|c1...cK) 
      float log_prob_wi_given_C = 0.0;
      // NOTE: since the center word is in the first position of prob_wi_z_given_C[idx], just used the idx
      for (int idx = 0; idx < local_embed_size_plus_one; idx++) log_prob_wi_given_C += prob_w_z_given_C[idx];
      if (log_prob_wi_given_C < 0.0000001){
	printf("WARNING: p(w|C) is very low: %f", log_prob_wi_given_C);
	fflush(stdout);
	log_prob_wi_given_C = log(log_prob_wi_given_C + epsilon);
      } else{
	log_prob_wi_given_C = log(log_prob_wi_given_C);
      }
      write_float(debug_file, "log p(w_i|C)", log_prob_wi_given_C, 0);

      float context_E_grad = 0.0;
      float center_word_E_grad = 0.0;
      float neg_center_word_E_grad = 0.0;

      // SUM OVER THE SAMPLED Z's
      // ONLY NEED TO CALC FOR PREDICTION PART OF GRAD
      write_str(debug_file, "PREDICTION GRADIENT");
      for (int m = 0; m < num_z_samples; m++) { 
	for (int k = 0; k < pos_context_counter; k++){
	  if (k == a) continue; // don't use center word w_i
	  context_word_position = pos_context_store[k] * embed_max_size;
	  for (int j = 0; j < z_samples[m]; j++){
	    context_E_grad = input_embed[center_word_position + j] - sparsity_weight*2*context_embed[context_word_position + j];
	    center_word_E_grad = context_embed[context_word_position + j] - sparsity_weight*2*input_embed[center_word_position + j];
	    gradient[k*local_embed_size_plus_one + j] += (1.0/num_z_samples) * ( -log_prob_wi_given_C ) * window_normalization * context_E_grad;
	    gradient[a*local_embed_size_plus_one + j] += (1.0/num_z_samples) * ( -log_prob_wi_given_C ) * window_normalization * center_word_E_grad;
	    write_float(debug_file, "context E grad", context_E_grad, j);
            write_float(debug_file, "center word E grad", center_word_E_grad, j);
            write_float(debug_file, "pos context gradient", gradient[k*local_embed_size_plus_one + j], j);
            write_float(debug_file, "dinput gradient", gradient[a*local_embed_size_plus_one + j], j);
          }
	}
      }

      // create variable that adjusts loops according to if dims were expanded 
      int loop_bound = local_embed_size_plus_one - 1;
      if (z_max == local_embed_size_plus_one){
	loop_bound = local_embed_size_plus_one;
      }

      // CALC DIMENSION GRADIENT TERM FOR CONTEXT & CENTER
      for (int k = 0; k < pos_context_counter; k++){
	if (k == a) continue;
	context_word_position = pos_context_store[k] * embed_max_size;
	for (int j = 0; j < loop_bound; j++){
	  context_E_grad = input_embed[center_word_position + j] - sparsity_weight*2*context_embed[context_word_position + j];
	  center_word_E_grad = context_embed[context_word_position + j] - sparsity_weight*2*input_embed[center_word_position + j]; 
	  gradient[k*local_embed_size_plus_one + j] += (log_prob_wi_given_C - 1) * sum_probs_z_given_w_C[j] * window_normalization * context_E_grad;
	  gradient[a*local_embed_size_plus_one + j] += (log_prob_wi_given_C - 1) * sum_probs_z_given_w_C[j] * window_normalization * center_word_E_grad;
	}
      }

      // CALC PREDICTION NORMALIZATION GRADIENT
      for (int j = 0; j < loop_bound; j++){
        for (int k = 0; k < pos_context_counter; k++){
	  if (k == a) continue;
	  context_word_position = pos_context_store[k] * embed_max_size;
	  // negative samples subgradient
	  for (d = 0; d < negative; d++){
	    neg_center_word_position = negative_list[d] * embed_max_size;
	    context_E_grad = input_embed[neg_center_word_position + j] - sparsity_weight*2*context_embed[context_word_position + j];
	    neg_center_word_E_grad = context_embed[context_word_position + j] - sparsity_weight*2*input_embed[neg_center_word_position + j]; 
	    gradient[k*local_embed_size_plus_one + j] += sum_prob_w_z_given_C[(d+1)*local_embed_size_plus_one + j] 
              * window_normalization * context_E_grad;
	    // add to gradient for negative example 
	    check_value((sum_prob_w_z_given_C[(d+1)*local_embed_size_plus_one + j] * neg_center_word_E_grad), "neg center word gradient", j);
	    neg_gradient[d*local_embed_size_plus_one + j] += sum_prob_w_z_given_C[(d+1)*local_embed_size_plus_one + j] * window_normalization * neg_center_word_E_grad;
	  }
	  // add subgradient for postive center word
	  center_word_E_grad = context_embed[context_word_position + j] - sparsity_weight*2*input_embed[center_word_position + j];
	  context_E_grad = input_embed[center_word_position + j] - sparsity_weight*2*context_embed[context_word_position + j]; 
	  // add to pos context gradient                                                                                                                                                  
	  gradient[k*local_embed_size_plus_one + j] += sum_prob_w_z_given_C[j] * window_normalization * context_E_grad;
	  // add to center word grad
	  gradient[a*local_embed_size_plus_one + j] += sum_prob_w_z_given_C[j] * window_normalization * center_word_E_grad;
	}
      }

      // MAKE FINAL GRAD UPDATES
      for (int j = 0; j < loop_bound; j++){
	for (int k = 0; k < pos_context_counter; k++){
	  context_word_position = pos_context_store[k] * embed_max_size;
	  check_value(gradient[k*local_embed_size_plus_one + j], "pos_context_gradient", j);
	  if (k == a){
	    input_embed[context_word_position + j] -= alpha_per_dim[j] * gradient[k*local_embed_size_plus_one + j];
	  }
	  else{
	    context_embed[context_word_position + j] -= alpha_per_dim[j] * gradient[k*local_embed_size_plus_one + j];
	  }
	}
        for (int d = 0; d < negative; d++) {
          neg_center_word_position = negative_list[d] * embed_max_size; 
          input_embed[neg_center_word_position + j] -= alpha_per_dim[j] * neg_gradient[d*local_embed_size_plus_one + j];
        }
      }

      // track training progress
      log_prob_per_word += -log_prob_wi_given_C;
      free(prob_w_z_given_C);
      free(sum_prob_w_z_given_C);
      free(gradient);
      free(neg_gradient);
    }
    // end loop over context (indexed by a)
    train_log_probability += log_prob_per_word/pos_context_counter;
    sentence_position++; 
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }

  fclose(fi);
  free(z_samples);   
  free(probs_z_given_w_C); 
  free(pos_context_store);
  free(sum_probs_z_given_w_C);
  free(negative_list); 
  
  fclose(debug_file);                                                           
  pthread_exit(NULL);
}

void TrainModel() {
  // Print start time
  char buff[100];                                                               
  time_t now = time (0);                                                          
  strftime(buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));               
  printf ("Strart training: %s\n", buff); 
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  // compute log of dim penalty
  log_dim_penalty = log(dim_penalty);
  // compute exp table
  build_exp_table(); 
 
  // expanded-dim training for desired epochs
  printf("Training expanded dim model for %lld iters \n", iter);
  
  thread_arg = malloc(sizeof(ThreadArg) * num_threads);
  for (int a = 0; a < num_threads; a++) {
    thread_arg[a].id = a;
    pthread_create(&pt[a], NULL, TrainModelThread, &thread_arg[a]);
  }
  free(thread_arg);
  for (int a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  printf("Writing input vectors to %s\n", output_file);
  save_vectors(output_file, vocab_size, embed_current_size, vocab, input_embed);
  printf("Writing context vectors to %s\n", context_output_file);
  if (strlen(context_output_file) > 0)  save_vectors(context_output_file, vocab_size, embed_current_size, vocab, context_embed);

  // free globally used space
  free(exp_table);
  if (adadelta_flag == 1){
    free(input_grad_history);
    free(context_grad_history);
  }
  free(alpha_count_adjustment);
  free(alpha_per_dim);
  free(input_embed);
  free(context_embed);
 
  // Print end time
  now = time (0);                                                               
  strftime(buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));               
  printf ("End Training: %s\n", buff);   
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
	printf("Argument missing for %s\n", str);
	exit(1);
      }
      return a;
    }
  return -1;
}

// testing function for sampling from multinomial
void multinom_unit_test(){
  // set up random number generator                                                                                                                                                                    
  const gsl_rng_type * T2;
  gsl_rng * r2;
  srand(time(NULL));
  unsigned int Seed2 = rand();
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r2 = gsl_rng_alloc (T2);
  gsl_rng_set (r2, Seed2);
  
  double x[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  for (int w=0; w<10; w++){
    int y = sample_from_mult(x, 6, r2);
    printf("Sampled idx: %i \n", y);
  }
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("INFINITE Word Embeddings\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting *input* word vectors\n");
    printf("\t-contextOutput <file>\n");
    printf("\t\tUse <file> to save the resulting *context* vectors\n");
    printf("\t-initSize <int>\n");
    printf("\t\tSet the initial dimensionality of the word vectors; default is 5\n");
    printf("\t-maxSize <int>\n");
    printf("\t\tSet the maximum dimensionality of the word vectors; default is 750\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words; Frequent ones will be downsampled.\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025.\n");
    printf("\t-dimPenalty <int>\n");
    printf("\t\tPenalty incurred for using each embedding dimension.  Must be in (1, infinity) to guarantee convergent Z. default=5.\n");
    printf("\t-sparsityWeight <float>\n");
    printf("\t\tWeight placed on L-2 sparsity penalty.  default = 0.001.\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-adaDelta <int>\n");
    printf("\t\tFlag that, if equal to one, sets the learning rate according to the AdaDelta adaptive gradient procedure.\n");
    printf("\t-epsilon <float>\n");
    printf("\t\tSmall value added to the denominator of the AdaDelta ratio for stability purposes.  Default: 0.1\n");
    printf("\t-rho <float>\n");
    printf("\t\tValue for AdaDelta's geometric average (rho)*grad_history + (1-rho)*grad^2.  Default: 0.95\n");
    printf("\nExamples:\n");
    printf("./iW2V -train data.txt -output w_vec.txt -contextOutput c_vec.txt -initSize 5 -maxSize 750 -window 5 -sample 1e-4 -negative 5 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-initSize", argc, argv)) > 0) embed_current_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-maxSize", argc, argv)) > 0) embed_max_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-dimPenalty", argc, argv)) > 0) dim_penalty = atof(argv[i+1]);
  if ((i = ArgPos((char *)"-sparsityWeight", argc, argv)) > 0) sparsity_weight = atof(argv[i+1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-contextOutput", argc, argv)) > 0) strcpy(context_output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-numSamples", argc, argv)) >0 ) num_z_samples = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-adaDelta", argc, argv)) >0 ) adadelta_flag = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-epsilon", argc, argv)) > 0) epsilon_adadelta = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) rho_adadelta = atof(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  print_args();
  TrainModel();
  return 0;
}
      
