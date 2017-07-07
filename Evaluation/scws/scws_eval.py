import sys
from Evaluation.eval_lib import read_embedding_file, dot_prod_sim, get_rank_corr, cosine_sim
from Evaluation.graph_p_z import compute_p_z_given_w
from Auto_Eval.auto_eval_iSG import compute_p_z_given_w_c 
import numpy as np
from math import exp, log
import re

SCWS_FILE = "Evaluation/scws/ratings.txt"

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

'''
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
'''

def expected_sim(w1, w2, p_z_given_w1, p_z_given_w2):
  sim = 0.
  for idx, x in enumerate(w1):
    sim += (p_z_given_w1[idx] + p_z_given_w2[idx]) * dot_prod_sim(w1[:idx+1],w2[:idx+1])

  return sim

def cosine_sim_main(embeddings, context_embeddings, w1_idx, w2_idx, c1_idx_arr, c2_idx_arr, sparsity, dim_penalty):
  w1 = embeddings[w1_idx]
  w2 = embeddings[w2_idx]
  return cosine_sim(w1, w2)

def p_z_w_c_sim(embeddings, context_embeddings, w1_idx, w2_idx, c1_idx_arr, c2_idx_arr, sparsity, dim_penalty):
  w1 = embeddings[w1_idx]
  c2 = context_embeddings[w2_idx]
  p_z_w_c = compute_p_z_given_w_c(w1, c2, sparsity, dim_penalty) 
  temp_sum = 0.0                                                              
  temp_prod = 0.0                                                             
  for dim_idx, z_weight in enumerate(p_z_w_c):                                
    temp_prod += w1[dim_idx]*c2[dim_idx]                      
    temp_sum += z_weight * temp_prod 
  
  return temp_sum

def p_z_w_sim(embeddings, context_embeddings, w1_idx, w2_idx, c1_idx_arr, c2_idx_arr, sparsity, dim_penalty):
  w1 = embeddings[w1_idx]                                                       
  w2 = embeddings[w2_idx] 
  c1 = map2embeddings(context_embeddings, c1_idx_arr)
  c2 = map2embeddings(context_embeddings, c2_idx_arr)
  p_z_given_w1 = compute_p_z_given_w(w1, c1, sparsity, dim_penalty)
  p_z_given_w2 = compute_p_z_given_w(w2, c2, sparsity, dim_penalty)
  return expected_sim(w1, w2, p_z_given_w1, p_z_given_w2)

def eval_scws(scws, embeddings, context_embeddings, sparsity, dim_penalty, sim_func):
  iw2v_sims = [] 
  for scores,w1_idx,w2_idx,c1_idx_arr,c2_idx_arr in scws:
    sim = sim_func(embeddings, context_embeddings, w1_idx, w2_idx, c1_idx_arr, c2_idx_arr, sparsity, dim_penalty)
    print vocab[w1_idx], vocab[w2_idx], sim
    iw2v_sims.append(sim)    

  return get_rank_corr(iw2v_sims)

if __name__ == '__main__':
  ### READ INPUT ARGS
  num_of_args = len(sys.argv)
  if num_of_args < 5:
    print "Usage: embedding_file context_embedding_file sparsity dim_penalty sim_idx "
    exit()
 
  embedding_filename = sys.argv[1]
  context_embedding_filename = sys.argv[2]
  sparsity = float(sys.argv[3])
  dim_penalty = float(sys.argv[4])

  print("Using embedding file: %s" % (embedding_filename))
  print("Using context embedding file: %s" % (context_embedding_filename))
  print("Sparsity: %f" % sparsity)
  print("Dim Penalty: %f" % dim_penalty)
  vocab, embeddings = read_embedding_file(embedding_filename)
  _, context_embeddings = read_embedding_file(context_embedding_filename)
  
  SIM_FUNCS = [cosine_sim_main, p_z_w_c_sim, p_z_w_sim]
  idx = int(sys.argv[5]) 
  if (idx >= 0 and idx <= 2):
    print "using sim function: ", SIM_FUNCS[idx]
  else:
    print "sim idx should be in {0,1,2}"
    exit()
  
  scws = read_scws(vocab)
  print("read scws data")
  corr = eval_scws(scws, embeddings, context_embeddings, sparsity, dim_penalty, SIM_FUNCS[idx])
  print corr
