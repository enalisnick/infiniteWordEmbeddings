import sys
from math import exp
import numpy as np
from scipy import spatial
from find_nearest_neighbors import dot_prod_sim

if __name__ == '__main__':
  ### READ INPUT ARGS
  num_of_args = len(sys.argv)
  if num_of_args < 2:
    print "Embedding file not specified...quitting."
    exit()
  embedding_fileName = sys.argv[1] 
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
  ### READ THROUGH AND COMPUTE FILE SIMS
  w2v_sims = []
  human_sims = []
  with open('Evaluation/wordSim353_sorted.csv') as f:
    for line in f:
      line = line.strip().split(',')
      try:
        word1_idx = vocab.index(line[0].lower())
        word2_idx = vocab.index(line[1].lower())
        human_sims.append(float(line[2]))
        word1_embedding = embeddings[word1_idx]
        word2_embedding = embeddings[word2_idx]
      ### compute how many dimensions to use
        z = 0
        max_prob = 0.0
        running_total = 0.
        for w,c in zip(word1_embedding[:z], word2_embedding[:z]):
          running_total += w*c 
        for z_idx in xrange(z,z_max):
          running_total += word1_embedding[z_idx]*word2_embedding[z_idx]
          if exp(running_total) > max_prob:
            z = z_idx+1
            max_prob = exp(running_total)
        #w2v_sims.append(z *(1 - spatial.distance.cosine(word1_embedding[:z], word2_embedding[:z])))
        w2v_sims.append(dot_prod_sim(word1_embedding[:z], word2_embedding[:z]))
      except ValueError:
        continue
### sort w2v sims
  sorted_w2v_idxs = sorted(range(len(w2v_sims)), key=w2v_sims.__getitem__)
  n = len(sorted_w2v_idxs)
  sum_squ_distances = 0.
  for idx, val in enumerate(sorted_w2v_idxs):
    sum_squ_distances += (idx-val)**2
  print 1.0 - (6*sum_squ_distances)/(n*(n**2 - 1.0))
