#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_randist.h>
#include "eval_lib.h"

#define MAX_SENTENCE_LENGTH 1000
#define MAX_STRING 100

#define STATUS_INTERVAL 5000

const int EXP_LEN = 100;
float *exp_table;
float dim_penalty, log_dim_penalty, sparsity_weight;

/*
  Build table which precompute exp function for certain integer
  values
*/
void build_exp_table() {
  exp_table = calloc(EXP_LEN * 2 + 1, sizeof(float));
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

// function to sample value of z_hat -- modified but essentially coppied from StackOverflow 
// http://stackoverflow.com/questions/25363450/generating-a-multinomial-distribution
int sample_from_mult(float float_probs[], int k, const gsl_rng* r){ // always sample 1 value
  double probs[k];
  for (int i = 0; i < k; i++) {
    probs[i] = float_probs[i];
  }
  unsigned int mult_op[k];
  gsl_ran_multinomial(r, k, 1, probs, mult_op);
  for (int idx=1; idx<=k; idx++){
    if (mult_op[idx-1]==1) return idx;
  }
  return 0;
}

// function to compute E(w, c, z)
float compute_energy(float *input_embed, float *context_embed, long long w_idx, 
  long long c_idx, int z, float log_dim_penalty, float sparsity_weight){
  long long a;
  float energy = 0.0;
  for (a = 0; a<z; a++) energy += 
    -input_embed[w_idx + a]*context_embed[c_idx + a] + log_dim_penalty 
    + sparsity_weight*input_embed[w_idx + a]*input_embed[w_idx+a] 
    + sparsity_weight*context_embed[c_idx + a]*context_embed[c_idx+a];
  return energy;
}

void compute_z_dist(float *dist, float *input_embed, float *context_embed, long long w_idx, 
  long long c_idx, int curr_z, float dim_penalty, float log_dim_penalty, float sparsity_weight) { 
  for (int a = 0; a < curr_z; a++) {
    float val = -input_embed[w_idx + a]*context_embed[c_idx + a] 
                +log_dim_penalty + sparsity_weight*input_embed[w_idx + a]*input_embed[w_idx + a] 
                +sparsity_weight*context_embed[c_idx + a]*context_embed[c_idx+a];
    for (int b = a; b <= curr_z; b++) {
      dist[b] += val;
    }
    dist[a] = exp_fast(-dist[a]);
    //partFunc += dist[a]; TODO: Eric added this line...do we need it?
  }
  dist[curr_z] = (dim_penalty / (dim_penalty - 1.0)) * exp_fast(-dist[curr_z]);
}
float get_log_prob(char *test_file_name, char *vocab, long long vocab_size, float *input_vectors, 
  float *context_vectors, long embed_size);

float get_log_prob_iw2v(long long vocab_size, float *input_embed, float *context_embed, 
  long long input_word_position, long long context_word_position, long embed_size, int num_negative);

// TODO: do this for w2v?
//float get_log_prob_w2v(long long vocab_size, float *input_embed, float *context_embed, long long input_word_position,
//long long context_word_position, long embed_size, int num_negative);
 
int main(int argc, char **argv) {
  char *vocab, *dummy_vocab; 
  char input_file_name[MAX_STRING], context_file_name[MAX_STRING], test_file_name[MAX_STRING];
  long long vocab_size, embed_size;
  float *input_vectors, *context_vectors; 
 
  // Build exp table
  build_exp_table(exp_table);
 
  strcpy(input_file_name, argv[1]);
  strcpy(context_file_name, argv[2]);
  strcpy(test_file_name, argv[3]);
  // TODO: get log_dim_penalty, dim_penalty & sparsity_weight values from file name 
  printf("Reading vectors...\n");
  read_vectors(input_file_name, &vocab_size, &embed_size, &vocab, &input_vectors);
  read_vectors(context_file_name, &vocab_size, &embed_size, &dummy_vocab, &context_vectors);

  printf("Starting testing...\n");
  float log_prob = get_log_prob(test_file_name, vocab, vocab_size, input_vectors, 
    context_vectors, embed_size);
  printf("-----------------------------------\n");
  printf("Final Log Probability: %f\n", log_prob);
  
  return 0;
}

float get_log_prob(char *test_file_name, char *vocab, long long vocab_size, float *input_vectors, 
  float *context_vectors, long embed_size) {
  int num_negative = 5;
  int window = 5; 
  long long last_word, word, sentence_length = 0, sentence_position = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long input_word_position, context_word_position;
  //float sample = 1e-3;
  //unsigned long long next_random = (long long) 1; 

  // Open corpus to read
  FILE *fi = fopen(test_file_name, "rb");
  fseek(fi, 0, SEEK_SET);
 
  float total_log_prob = 0.0;
  long iter = 0;
  while (1) {  
    // read a new sentence / line
    if (sentence_length == 0) {
      while (1) {
	//word = ReadWordIndex(fi);
        char word_str[MAX_STRING];	
        ReadWord(word_str, fi);
        word = find_str(vocab, vocab_size, word_str); 
        if (feof(fi)) break;
	if (word == -1) continue;
	if (word == 0) break;
	// The subsampling randomly discards frequent words while keeping the ranking same
	// NOTE: not doing this because need to keep track of count of words
        /*if (sample > 0) {
	  double ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
	  next_random = next_random * (unsigned long long)25214903917 + 11;
	  if (ran < (next_random & 0xFFFF) / (double)65536) continue;
	}*/
	sen[sentence_length] = word;
	sentence_length++;
	if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    // if EOF, break
    if (feof(fi) ) break;
 
    // start of test, get current word (w)
    word = sen[sentence_position];
    input_word_position = word * embed_size; 
    
    // iterate through the context words
    float sum_log_prob = 0.0;
    for (int a = 0; a < window * 2 + 1; a++) if (a != window) {
      int c = sentence_position - window + a;
      if (c < 0) continue;
      if (c >= sentence_length) continue;
      last_word = sen[c];
      if (last_word == -1) continue;
      
      context_word_position = last_word * embed_size;
      float prob = get_log_prob_iw2v(vocab_size, input_vectors, context_vectors, input_word_position, 
        context_word_position, embed_size, num_negative); 
      sum_log_prob += prob;
    }
    total_log_prob += sum_log_prob;
    iter++;
    if (iter % STATUS_INTERVAL == 0)  printf("Iteration: %ld, Log Probability: %f\n", iter, 
                                             total_log_prob/iter);
 
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    } 
  }    

  return (total_log_prob/iter);
}

float get_log_prob_iw2v(long long vocab_size, float *input_embed, float *context_embed, long long input_word_position, long long context_word_position, long embed_size, int num_negative) {
 
  // set up random number generator
  const gsl_rng_type * T2;
  gsl_rng * r2;
  srand(time(NULL)); 
  unsigned int Seed2 = rand();
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r2 = gsl_rng_alloc (T2);
  gsl_rng_set (r2, Seed2);
 
  float *z_probs = (float *)calloc(embed_size, sizeof(float));
  compute_z_dist(z_probs, input_embed, context_embed, input_word_position, context_word_position, 
    embed_size, dim_penalty, log_dim_penalty, sparsity_weight); 
  int z_hat = sample_from_mult(z_probs, embed_size, r2);  // only 1 sample 

  long long negative_word, negative_word_position;
  float neg_probs[num_negative];
  float Z_c = 0;
  for (int d = 0; d < num_negative; d++) {
    negative_word = (long) (((float)vocab_size)/RAND_MAX * rand()); 
    negative_word_position = negative_word * embed_size;
    neg_probs[d] = exp_fast(-compute_energy(input_embed, context_embed, input_word_position, 
                            negative_word_position, z_hat, log_dim_penalty, sparsity_weight)); 
    Z_c += neg_probs[d];
  }

  float pos_prob = exp_fast(-compute_energy(input_embed, context_embed, input_word_position, 
                            context_word_position, z_hat, log_dim_penalty, sparsity_weight));
  Z_c += pos_prob;
  free(z_probs);
  return log(pos_prob/Z_c);
}

// TODO: do for w2v?
/*float get_log_prob_w2v(long long vocab_size, float *input_embed, float *context_embed, long long input_word_position,   long long context_word_position, long embed_size, int num_negative) {

  return log(sigmoid(input_embed, context_embed, embed_size, input_word_position, context_word_position)); 
}*/
