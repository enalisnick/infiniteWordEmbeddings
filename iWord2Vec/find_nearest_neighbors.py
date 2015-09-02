from scipy import spatial
from tabulate import tabulate

### READ EMBEDDINGS FROM TXT FILE
embedding_fileName = "output.txt"
embeddings = []
vocab = []
with open(embedding_fileName) as f:
  for line in f.readlines()[1:]:
    line = line.strip().split()
    vocab.append(line[0])
    embeddings.append([float(x) for x in line[1:]])
vocab_size = len(vocab)

headers = ["Rank", "Word", "Cosine Similarity"]
k = 50 # number of nearest neighbors
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
        sims[idx] = 1 - spatial.distance.cosine(embeddings[word_idx], embeddings[idx])
    ### get top k most similar 
    top_k_idxs = sorted(range(vocab_size), key=sims.__getitem__, reverse=True)[:k]
    t = []
    print "%d nearest neighbors to %s:" %(k, input)
    for i, idx in enumerate(top_k_idxs):
      t.append([i+1, vocab[idx], sims[idx]])
    # output table
    print tabulate(t,headers=headers)
  except ValueError:
    print "There's no embedding for that word.  Try again."
