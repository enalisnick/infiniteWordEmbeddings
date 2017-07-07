#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_randist.h>
#include "../Evaluation/eval_lib.h"

#define MAX_SENTENCE_LENGTH 1000
#define MAX_STRING 100

#define STATUS_INTERVAL 15
const int EXP_LEN = 87;
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const double epsilon = 1e-8;

struct vocab_word {
  long long cn;
  char *word;
};
struct vocab_word *vocab;
int min_count = 25;
float *exp_table;
float dim_penalty, log_dim_penalty, sparsity_weight;
float *input_embed, *context_embed;
long long embed_size, train_words;
const int table_size = 1e8;
long long vocab_max_size = 1000, vocab_size = 0;
int *table;
int *vocab_hash;
char read_vocab_file[MAX_STRING];
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

/********************************************
  iCBOW Probability Functions (copied over but
  edited to not have infinte part)
********************************************/
float compute_z_dist(float *dist, long long *context, int center_idx, int context_size, int embed_size) { 
  long long w_idx = context[center_idx] * embed_size;
  float window_norm = 1.0/(context_size - 1.0);
  float max_value = 0.0;
  for (int a = 0; a < embed_size; a++) {
    // precompute context values
    float context_sum = 0;
    float context_norms = 0;
    for (int j = 0; j < context_size; j++){
      if (j == center_idx) continue;
      long long c_idx = context[j] * embed_size;
      context_sum += context_embed[c_idx + a];
      context_norms += context_embed[c_idx + a] * context_embed[c_idx + a];
    }
    // compute entergy
    float val = -window_norm*input_embed[w_idx + a]*context_sum 
      + log_dim_penalty + sparsity_weight*input_embed[w_idx + a]*input_embed[w_idx + a] 
                + window_norm*sparsity_weight*context_norms;
    for (int b = a; b < embed_size; b++) {
      dist[b] += val;
    }
    if (-dist[a] > max_value) max_value = -dist[a];
  }

  return max_value;
}

void compute_p_z_given_w_C(float *prob_z_given_w_C, float *sum_prob_z_given_w_C, long long *context, 
  int center_idx, int context_size, int embed_size) {
  float norm = 0.0;
  float max_value = compute_z_dist(prob_z_given_w_C, context, center_idx, context_size, embed_size);
  // now exponentiate and normalize                                              
  for (int a = 0; a < embed_size; a++) {                    
    prob_z_given_w_C[a] = exp_fast(-prob_z_given_w_C[a]-max_value);          
    norm += prob_z_given_w_C[a];            
  }                   

  // pre-calculate sums
  float sum = 0.0;
  for (int a = embed_size - 1; a >= 0; a--) {
    prob_z_given_w_C[a] = prob_z_given_w_C[a]/norm;
    sum += prob_z_given_w_C[a];
    sum_prob_z_given_w_C[a] = sum;
  }
}

// prob_c_z_given_w should be of size true_context_size * curr_z_plus_one
//(a, pos_context_store, negative_list, pos_context_counter, negative, prob_w_z_given_C, local_embed_size_plus_one)
void compute_p_w_z_given_C(long long center_idx, long long *context, long long *negatives, int context_size, int negative_size, float *prob_w_z_given_C, float *sum_prob_w_z_given_C, int embed_size) {
  // compute e^(-E(w,c,z)) for z = 1,...,curr_z,curr_z+1 for every context c
  float norm = 0.0;
  long long save_true_w = context[center_idx];
  float max_value = 0.0;
  float temp_value = 0.0;

  // iterate through once to compute energies and find max value
  max_value = compute_z_dist(prob_w_z_given_C, context, center_idx, context_size, embed_size);
  for (int s = 0; s < negative_size; s++) {
    context[center_idx] = negatives[s];
    temp_value = compute_z_dist(prob_w_z_given_C + (s+1) * embed_size, context, center_idx, context_size, embed_size);
    if (temp_value > max_value) max_value = temp_value;
  }
  // replace true center word
  context[center_idx] = save_true_w;

  // now iterate through again to exponentiate and compute norm
  for (int s = 0; s < negative_size + 1; s++){
    for (int j = 0; j < embed_size; j++){
      prob_w_z_given_C[s*embed_size + j] = exp_fast(-prob_w_z_given_C[s*embed_size + j]-max_value);
      norm += prob_w_z_given_C[s*embed_size + j];
    }
  }
  
  // compute prob
  for (int s = 0; s < negative_size + 1; s++) {
    float sum = 0.0;
    for (int z = embed_size-1; z >= 0; z--) {
      prob_w_z_given_C[s * embed_size + z] = prob_w_z_given_C[s * embed_size + z]/norm;
      sum += prob_w_z_given_C[s * embed_size + z];
      sum_prob_w_z_given_C[s * embed_size + z] = sum;
    }
  }
}
/********************************************
********************************************/

float get_log_prob(char *test_file_name, float *input_embed, float *context_embed, long embed_size) {
  int negative = 5;
  int window = 5; 
  unsigned long long next_random = 1;
  long long a, b, d, c, word, last_word, negative_word, sentence_length = 0, sentence_position = 0;
  long long sen[MAX_SENTENCE_LENGTH + 1];
  long long input_word_position, center_word;
  
  // terms needed for p(z|w,C)
  float *prob_z_given_w_C = (float *) calloc(embed_size, sizeof(float));
  float *sum_prob_z_given_w_C = (float *) calloc(embed_size, sizeof(float));
  // terms needed for p(w,z|C)
  float *prob_w_z_given_C = (float *) calloc(embed_size * (negative + 1), sizeof(float));
  float *sum_prob_w_z_given_C = (float *) calloc(embed_size * (negative + 1), sizeof(float));   
  // stores negative constext
  long long *neg_context = (long long *) calloc(negative, sizeof(long long)); 
  // stores positive context
  long long *pos_context = (long long *) calloc(2*window+1, sizeof(long long));

  // Open corpus to read
  FILE *fi = fopen(test_file_name, "rb");
  fseek(fi, 0, SEEK_SET);
 
  float total_log_prob = 0.0;
  long iter = 0;
  while (1) {  
    // read a new sentence / line
    if (sentence_length == 0) {
      while (1) {
	word = ReadWordIndex(fi);
        if (feof(fi)) break;
	if (word == -1) continue;
	if (word == 0) break;
	sen[sentence_length] = word;
	sentence_length++;
	if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    // if EOF, break
    if ( sentence_length==0 && feof(fi) ) break;
 
    // start of test, get current word (w)
    word = sen[sentence_position];
    input_word_position = word * embed_size; 
    
    // MAIN LOOP THROUGH POSITIVE CONTEXT
    int pos_context_counter = 0; // size of positive context
    b = 0;
    float log_prob_current_context = 0.0;
    input_word_position = 0;
    for (a = b; a < window * 2 + 1 - b; a++){
	c = sentence_position - window + a;
	if (c < 0) continue;
	if (c >= sentence_length) continue;
	last_word = sen[c];
	if (last_word <= 0) continue;
	pos_context[pos_context_counter] = last_word;
	if (c == sentence_position) input_word_position = pos_context_counter;
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

    // center word to predict
    center_word = pos_context[input_word_position];
      
    // NEGATIVE SAMPLING CONTEXT WORDS
    d = negative-1;
    while (d >= 0) {
      neg_context[d] = 0; // clear old contexts
      next_random = next_random * (unsigned long long)25214903917 + 11;
      negative_word = table[(next_random >> 16) % table_size];
      if (negative_word == 0) negative_word = next_random % (vocab_size - 1) + 1;
      if (negative_word == center_word || negative_word <= 0) continue; 
      neg_context[d] = negative_word;
      d--;
    }

    // CALCULATE NECESSARY PROBABILITIES 
    // compute p(w,z|C)
    compute_p_w_z_given_C(input_word_position, pos_context, neg_context, pos_context_counter, 
      negative, prob_w_z_given_C, sum_prob_w_z_given_C, embed_size);

    float prob_w_C = sum_prob_w_z_given_C[0];
    log_prob_current_context += log(prob_w_C + epsilon); 
    total_log_prob += log_prob_current_context;
    iter++;
    
    if (iter % STATUS_INTERVAL == 0) printf("Iteration: %ld, Perplexity: %f\n", iter, exp(-total_log_prob/iter));
    fflush(stdout);

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    } 
  }

  free(pos_context);
  free(neg_context);
  free(prob_z_given_w_C);
  free(sum_prob_z_given_w_C);
  free(prob_w_z_given_C);
  free(sum_prob_w_z_given_C);
  return exp(-total_log_prob/iter);
}

int main(int argc, char **argv) {
  char *vocab_local, *dummy_vocab;
  long long vocab_size_local = 0;
  char input_file_name[MAX_STRING], context_file_name[MAX_STRING], test_file_name[MAX_STRING];

  // Build exp table                                                                                                                                           
  build_exp_table();
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  // Read arguments from command line                                                                                                                                                                                            
  strcpy(input_file_name, argv[1]);
  strcpy(context_file_name, argv[2]);
  strcpy(test_file_name, argv[3]);
  strcpy(read_vocab_file, argv[4]);
  sparsity_weight = atof(argv[5]);
  dim_penalty = atof(argv[6]);
  log_dim_penalty = log(dim_penalty);
  // log what we read in                                                                                                                                                                                           
  printf("%s\n", input_file_name);
  read_vectors(input_file_name, &vocab_size_local, &embed_size, &vocab_local, &input_embed);
  read_vectors(context_file_name, &vocab_size_local, &embed_size, &dummy_vocab, &context_embed);

  ReadVocab();
  InitUnigramTable();

  printf("Input vectors: %s\n", input_file_name);
  printf("Context vectors: %s\n", context_file_name);
  printf("Sparsity weight: %.8f\n", sparsity_weight);
  printf("Dimension penalty: %.8f\n", dim_penalty);
  printf("Embedding size: %lld\n", embed_size);
  fflush(stdout);
  
  printf("Starting testing...\n");
  float log_prob = get_log_prob(test_file_name, input_embed, context_embed, embed_size);
  free(exp_table);
  free(vocab);
  free(vocab_hash);
  printf("-----------------------------------\n");
  printf("Final Perplexity: %f\n", log_prob);
  fflush(stdout);
  return 0;
}

