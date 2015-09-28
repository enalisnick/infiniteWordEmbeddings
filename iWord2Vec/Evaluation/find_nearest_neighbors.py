import sys
from math import exp
import numpy as np
from scipy import spatial
from tabulate import tabulate
import math

# dot product similarity
def dot_prod_sim(v1,v2):
  total = 0.
  for w,c in zip(v1,v2):
    total += w*c 

  return total

if __name__ == '__main__':
  
  ### READ INPUT ARGS
  num_of_args = len(sys.argv)
  if num_of_args < 2:
    print "Embedding file not specified...quitting."
    exit()

  k = 50
  num_dims = -1
  embedding_fileName = sys.argv[1]
  if num_of_args >= 3:
    k = int(sys.argv[2]) # number of nearest neighbors
  if num_of_args >= 4:
    num_dims = int(sys.argv[3]) # number of dimensions to do similarity over
   
  ### READ EMBEDDINGS FROM TXT FILE
  embeddings = []
  vocab = []
  with open(embedding_fileName) as f:
    for line in f.readlines()[1:]:
      line = line.strip().split()
      vocab.append(line[0])
      embeddings.append([float(x) for x in line[1:]])
  vocab_size = len(vocab)
  z_max = len(embeddings[0])
  headers = ["Rank", "Word", "Cosine Similarity", "Num. of Dimensions Used"]
  input = "none"
  while True:
    input = raw_input("Enter word or 'EXIT' to quit: ").split()[0]
    if input=="EXIT":
      break
    word_idx = -1
    try:
      word_idx = vocab.index(input)
      ### compute cosine similarity with all words
      sims = [-1.]*vocab_size
      z_vals = [0]*vocab_size
      for idx in xrange(vocab_size):
        if idx != word_idx:
          word_embedding = embeddings[word_idx]
          other_word_embedding = embeddings[idx]
          ### compute how many dimensions to use
          z = 0
          max_prob = 0.0
          running_total = 0.
          for w,c in zip(word_embedding[:z], other_word_embedding[:z]):
            running_total += w*c 
          for z_idx in xrange(z,z_max):
            running_total += word_embedding[z_idx]*other_word_embedding[z_idx]
            if exp(running_total) > max_prob:
              z = z_idx+1
              max_prob = exp(running_total)
          #sims[idx] = z * (1 - spatial.distance.cosine(word_embedding[:z], other_word_embedding[:z]))
          sims[idx] = dot_prod_sim(word_embedding[:z], other_word_embedding[:z])
          z_vals[idx] = z
      ### get top k most similar 
      top_k_idxs = sorted(range(vocab_size), key=sims.__getitem__, reverse=True)[:k]
      t = []
      print "%d nearest neighbors to %s:" %(k, input)
      for i, idx in enumerate(top_k_idxs):
        t.append([i+1, vocab[idx], sims[idx], z_vals[idx]])
      # output table
      print tabulate(t,headers=headers)
      #print(embeddings[word_idx])
      #print("#non-zero dims: %d" % (np.count_nonzero(embeddings[word_idx])))
    except ValueError:
      print "There's no embedding for that word.  Try again."
