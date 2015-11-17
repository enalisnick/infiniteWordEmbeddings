from math import exp, log
from scipy import spatial

### cosine similarity
def cosine_sim(v1,v2):
  return (1.0 - spatial.distance.cosine(v1,v2))

### dot product similarity
def dot_prod_sim(v1,v2):
  total = 0.
  for w,c in zip(v1,v2):
    total += w*c  

  return total

### get z according to mode of p(z | w,c)
def get_mode_z(v1,v2):
  z_max = len(v1)
  z = 0
  max_prob = 0.0
  running_total = 0.
  for z_idx in xrange(z,z_max):
    running_total += v1[z_idx]*v2[z_idx]
    if exp(running_total) > max_prob:
      z = z_idx+1
      max_prob = exp(running_total)
  
  return z

### get z according to mode of p(z | w,c)
def get_mode_z_context(v1,c2):
  sparsity_weight = 0.001                                                     
  dim_penalty = 1.1
  z_max = len(v1)                                                               
  z = 0                                                                         
  max_prob = 0.0                                                                
  running_total = 0.                                                                                                                    
  for z_idx in xrange(z,z_max):                                                 
    running_total += v1[z_idx]*c2[z_idx] - log(dim_penalty) - sparsity_weight*pow(v1[z_idx],2) - sparsity_weight*pow(c2[z_idx],2)                                      
    exp_val = exp(running_total)
    if exp_val > max_prob:                                           
      z = z_idx+1                                                               
      max_prob = exp_val                                             
                                                                                
  return z                         

### get vocab and word-embeddings from file 
def read_embedding_file(embedding_filename):
  ### READ EMBEDDINGS FROM TXT FILE
  embeddings = []
  vocab = []
  with open(embedding_filename) as f:
    for line in f.readlines()[1:]:
      line = line.strip().split()
      vocab.append(line[0])
      embeddings.append([float(x) for x in line[1:]])
 
  return vocab, embeddings  

### get K nearest neighbors 
def get_nn(vocab, embeddings, context_embeddings, word, K, num_dims=-1):
  vocab_size = len(vocab)
  word_idx = vocab.index(word)
  
  ### compute cosine similarity with all words
  sims = [-1.]*vocab_size
  z_vals = [0]*vocab_size
  for idx in xrange(vocab_size):
    if idx != word_idx:
      word_embedding = embeddings[word_idx]
      other_word_embedding = context_embeddings[idx]
       
      ### compute how many dimensions to use
      z = 0
      if num_dims > 0:
	z = num_dims
      else:
	z = get_mode_z_context(word_embedding, other_word_embedding)
              
      sims[idx] = dot_prod_sim(word_embedding[:z], embeddings[idx][:z])
      z_vals[idx] = z
  ### get top k most similar 
  top_k_idxs = sorted(range(vocab_size), key=sims.__getitem__, reverse=True)[:K]
  return sims, z_vals, top_k_idxs 


def get_rank_corr(w2v_sims):
  ### sort w2v sims
  sorted_w2v_idxs = sorted(range(len(w2v_sims)), key=w2v_sims.__getitem__)
  n = len(sorted_w2v_idxs)
  sum_squ_distances = 0.
  for idx, val in enumerate(sorted_w2v_idxs):
    sum_squ_distances += (idx-val)**2
  
  return (1.0 - (6*sum_squ_distances)/(n*(n**2 - 1.0)))

'''
  Compute pearson rank correlation for similarity file given.
  Using input & context for mode_z calculation and input & input dot-product
  for similarity.
'''
def get_rank_corr_for_sim(sim_file, vocab, embeddings, context_embeddings, full_dim):
  w2v_sims = []
  human_sims = []

  with open(sim_file) as f:
    for line in f:
      if ',' in line:
        line = line.strip().split(',')  # for sim353
      else:
        line = line.strip().split() # for MEN
      try:
        word1_idx = vocab.index(line[0].lower())
        word2_idx = vocab.index(line[1].lower())
        human_sims.append(float(line[2]))
        word1_embedding = embeddings[word1_idx]
        if full_dim:
          word2_embedding = embeddings[word2_idx]
          w2v_sims.append(cosine_sim(word1_embedding, word2_embedding))
        else:
          ### compute how many dimensions to use
          word2_embedding = context_embeddings[word2_idx]
          z = get_mode_z_context(word1_embedding, word2_embedding)
          word2_embedding = embeddings[word2_idx]
          sim = dot_prod_sim(word1_embedding[:z], word2_embedding[:z])
          w2v_sims.append(sim)
      except ValueError:
        continue
  
  return get_rank_corr(w2v_sims) 
