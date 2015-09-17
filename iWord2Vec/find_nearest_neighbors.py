import sys
import numpy as np
from scipy import spatial
from tabulate import tabulate

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

  headers = ["Rank", "Word", "Cosine Similarity"]
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
      for idx in xrange(vocab_size):
        if idx != word_idx:
          word_embedding = embeddings[word_idx]
          other_word_embedding = embeddings[idx]
          if num_dims > 0 and num_dims <= len(embeddings[word_idx]):
            word_embedding = word_embedding[0:num_dims]
            other_word_embedding = other_word_embedding[0:num_dims]  
              
          sims[idx] = 1 - spatial.distance.cosine(word_embedding, other_word_embedding)

      ### get top k most similar 
      top_k_idxs = sorted(range(vocab_size), key=sims.__getitem__, reverse=True)[:k]
      t = []
      print "%d nearest neighbors to %s:" %(k, input)
      for i, idx in enumerate(top_k_idxs):
        t.append([i+1, vocab[idx], sims[idx]])
      # output table
      print tabulate(t,headers=headers)
      print(embeddings[word_idx])
      print("#non-zero dims: %d" % (np.count_nonzero(embeddings[word_idx])))
    except ValueError:
      print "There's no embedding for that word.  Try again."

