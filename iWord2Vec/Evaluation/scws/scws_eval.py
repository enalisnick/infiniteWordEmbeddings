import sys
from Evaluation.eval_lib import read_embedding_file, get_mode_z, get_mode_z_context, get_nn, dot_prod_sim, get_rank_corr, cosine_sim
from Evaluation.graph_p_z import compute_p_z_given_w
import numpy as np
from math import exp, log
import re

SCWS_FILE = "Evaluation/scws/ratings.txt"

COSINE = "cosine"
DOT_PROD = "dot_product"
EXP_SIM = "exp_sim"

# get mean embedding from set of word indices 
def get_mean_vector(embeddings, c_arr):
  arr_sum = np.zeros(len(embeddings[0]))  
  for c in c_arr:
    arr_sum = arr_sum + embeddings[c]
  
  arr_sum /= len(c_arr)
  return arr_sum

# map list of indices to list of embeddings
def map2embeddings(embeddings, c_arr):
  arr = []
  for c in c_arr:
    arr.append(embeddings[c])
  
  return arr

# map list of word strings to list of indices
def map2vocab(vocab, c_arr):
  arr = []
  for c in c_arr:
    try: 
      d = vocab.index(c.lower())
      arr.append(d)
    except ValueError:
      pass 
  return arr

'''
  Read Scws file and return list of (mean_rating, word w1, word w2, contexts c_11,...,c_1n, contexts c_21,...,c_2n)  
  examples from dataset.
  Borrowed from: https://github.com/srifai/senna/blob/master/aistats/testscripts/scws/test.py
'''
def read_scws(vocab, scws_file=SCWS_FILE, amount=-1):
  text = open(scws_file, 'r')

  scws = []
  cnt = 0
  for i,d in enumerate(text):
    split1 = d.lower().split('\t')
    idx = int(split1[0])
    w1 = split1[1]
    w2 = split1[3]
    scores = [ float(dd) for dd in split1[-11:] ][1:]
    c1,c2 = split1[5:-11]

    '''
    # only use sentence in which word exists
    search_str_period = "\\..*?<b>.*</b>.*?\\."
    search_str_no_period_right = "\\..*?<b>.*</b>.*?"
    search_str_no_period_left = ".*?<b>.*</b>.*?\\."
    search_str_no_period = ".*?<b>.*</b>.*?"
    search_strs = [search_str_period, search_str_no_period_right, 
     search_str_no_period_left, search_str_no_period]
    for search_str in search_strs:
      search1 = re.search(search_str, c1) 
      if search1 is not None:
        c1 = search1.group(0) 
        break
    for search_str in search_strs:
      search1 = re.search(search_str, c2)
      if search1 is not None:
        c2 = search1.group(0)
        break  
    '''
    c1 = [ w.replace('<b>','') for w in c1.split(' ') ]
    c2 = [ w.replace('<b>','') for w in c2.split(' ') ]

    w1 = [ w for j,w in enumerate(c1[:-1]) if w1 == w and c1[j+1] == '</b>' ][0]
    w2 = [ w for j,w in enumerate(c2[:-1]) if w2 == w and c2[j+1] == '</b>' ][0]

    c1 = [ w for w in c1 if w != '</b>' ]
    c2 = [ w for w in c2 if w != '</b>' ]

    c1_idxs = map2vocab(vocab,c1)
    c2_idxs = map2vocab(vocab,c2) 
    try:
      print w1.lower(), w2.lower()
      element = (sum(scores)/float(len(scores)),vocab.index(w1.lower()),vocab.index(w2.lower()),c1_idxs,c2_idxs)
      scws.append(element) 
    except ValueError:
      pass
    
    cnt += 1  
    if amount > 0 and cnt > amount: break

  # sort in ascending order 
  return sorted(scws, key=lambda x : x[0])

def expected_sim(w1, w2, p_z_given_w1, p_z_given_w2):
  sim = 0.0
  for idx, x in enumerate(w1):
      if p_z_given_w1[idx] > 0.00001 and p_z_given_w2[idx] > 0.00001:
        sim += p_z_given_w1[idx] * p_z_given_w2[idx] * dot_prod_sim(w1[:idx+1],w2[:idx+1])
 
  return sim

def expected_sim2(w1, w2, p_z_w1, p_z_w2):
  sim = 0.0
  for idx, x in enumerate(w1):
    for idx2, y in enumerate(w2):
      i = min(idx,idx2)
      if p_z_w1[idx] > 0.00001 and p_z_w2[idx2] > 0.00001:
        sim +=  p_z_w1[idx] * p_z_w2[idx2] * dot_prod_sim(w1[:i+1],w2[:i+1])
 
  return sim

def eval_scws(vocab, embeddings, context_embeddings, scws, arg_hash):
  iw2v_sims = [] 
  for scores,w1_idx,w2_idx,c1_idx_arr,c2_idx_arr in scws:
    w1 = embeddings[w1_idx]
    w2 = embeddings[w2_idx]
        
    sim = 0.
    if arg_hash[COSINE] == True:
      sim = cosine_sim(w1, w2)
    elif arg_hash[DOT_PROD] == True:
      c2 = context_embeddings[w2_idx]
      z = get_mode_z_context(w1, c2)
      sim = dot_prod_sim(w1[:z], w2[:z])
    elif arg_hash[EXP_SIM] == True: 
      p_z_given_w1 = compute_p_z_given_w(w1, map2embeddings(context_embeddings, c1_idx_arr)) 
      p_z_given_w2 = compute_p_z_given_w(w2, map2embeddings(context_embeddings, c2_idx_arr))
      sim = expected_sim2(w1,w2,p_z_given_w1,p_z_given_w2) 
      
    print vocab[w1_idx], vocab[w2_idx], sim
    iw2v_sims.append(sim)    

  return get_rank_corr(iw2v_sims)

if __name__ == '__main__':
  ### READ INPUT ARGS
  use_full_dim = 0
  num_of_args = len(sys.argv)
  if num_of_args < 2:
    print "Embedding file not specified...quitting."
    exit()
 
  embedding_filename = sys.argv[1]
  context_embedding_filename = sys.argv[2]
  print("Using embedding file: %s" % (embedding_filename))
  vocab, embeddings = read_embedding_file(embedding_filename)
  _, context_embeddings = read_embedding_file(context_embedding_filename)
  ### Read type of sim to use
  # 0. Cosine similarity
  # 1. Dot product similarity ~ using mode z
  # 2. Expected similarity using dot product
  arg_hash = {COSINE:False, DOT_PROD:False, EXP_SIM:False}
  arg = int(sys.argv[3]) 
  if arg == 0:
    arg_hash[COSINE] = True 
  elif arg == 1:
    arg_hash[DOT_PROD] = True 
  elif arg == 2:
    arg_hash[EXP_SIM] = True 
  print 'arg hash: ', arg_hash

  scws = read_scws(vocab)
  print("read scws data")
  corr = eval_scws(vocab, embeddings, context_embeddings, scws, arg_hash)
  print corr
