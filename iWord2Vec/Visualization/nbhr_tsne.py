import sys
import numpy as np
from sklearn.manifold import TSNE 
import matplotlib as mpl
mpl.use('Agg') # so that we can save plot without displaying
import matplotlib.pyplot as plt

from Evaluation.eval_lib import read_embedding_file  
from Evaluation.eval_lib import get_nn

if __name__ == '__main__':
  embedding_filename = sys.argv[1]
  word = sys.argv[2]
  K = int(sys.argv[3])
  num_dims = int(sys.argv[4]) # number of dimensions to do search over
  path = sys.argv[5] # path where to write plot
  
  vocab, embeddings = read_embedding_file(embedding_filename)
  dim = len(embeddings[0])
  sims, z_vals, top_k_idxs = get_nn(vocab, embeddings, word, K, num_dims)

  ### perform t-SNE   
  X= np.zeros(shape=(K+1, dim))
  X[0,] = embeddings[vocab.index(word)][:num_dims]
  for i, idx in enumerate(top_k_idxs):
    X[i+1,] = embeddings[idx][:num_dims]

  model = TSNE(n_components=2, random_state=0)
  Y = model.fit_transform(X)

  ### set up plot with title and label for points 
  plt.figure(figsize=(15,15))
  plt.scatter(Y[:,0], Y[:,1])
  plt.title("Nearest neighbors for '" + word + "' restricted to " + str(num_dims) + " dims")
 
  plt.annotate(word, xy = (Y[0,0], Y[0,1]), xytext = (0, 0),
        textcoords = 'offset points', ha = 'right', va = 'bottom') 
  for idx, x, y in zip(top_k_idxs, Y[1:, 0], Y[1:, 1]):
    plt.annotate(
        vocab[idx], 
        xy = (x, y), xytext = (0, 0),
        textcoords = 'offset points', ha = 'right', va = 'bottom')
  
  plot_name = word + '_K=' + str(K) + '_dims=' + str(num_dims)
  plt.savefig(path + plot_name + ".jpg")
