import sys
from math import exp
import numpy as np
from scipy import spatial
from tabulate import tabulate
import math

from Evaluation.eval_lib import read_embedding_file, get_nn

if __name__ == '__main__':
  
  ### READ INPUT ARGS
  num_of_args = len(sys.argv)
  if num_of_args < 3:
    print "Embedding & Context file not specified...quitting."
    exit()

  k = 50
  num_dims = -1
  embedding_filename = sys.argv[1]
  context_embedding_filename = sys.argv[2]
  if num_of_args >= 4:
    k = int(sys.argv[3]) # number of nearest neighbors 
  if num_of_args >= 5:
    num_dims = int(sys.argv[4]) # number of dimensions to do similarity over
  
  vocab, embeddings = read_embedding_file(embedding_filename)
  _, context_embeddings = read_embedding_file(context_embedding_filename)
  
  headers = ["Rank", "Word", "Cosine Similarity", "Num. of Dimensions Used"]
  input = "none"
  while True:
    input = raw_input("Enter word or 'EXIT' to quit: ").split()[0]
    if input=="EXIT":
      break
    word_idx = -1
    try:
      word_idx = vocab.index(input)
      sims, z_vals, top_k_idxs = get_nn(vocab, embeddings, context_embeddings, vocab[word_idx], k, num_dims)   
      t = []
      print "%d nearest neighbors to %s:" %(k, input)
      for i, idx in enumerate(top_k_idxs):
        t.append([i+1, vocab[idx], sims[idx], z_vals[idx]])
      # output table
      print tabulate(t,headers=headers)
      print(embeddings[word_idx])
      print("#non-zero dims: %d" % (np.count_nonzero(embeddings[word_idx])))
    except ValueError:
      print "There's no embedding for that word.  Try again."
