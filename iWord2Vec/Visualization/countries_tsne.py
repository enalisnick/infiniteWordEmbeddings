import sys
import numpy as np
from sklearn.manifold import TSNE 
import matplotlib as mpl
mpl.use('Agg') # so that we can save plot without displaying
import matplotlib.pyplot as plt
from random import randint

from Evaluation.eval_lib import read_embedding_file  
from Evaluation.eval_lib import get_nn

countries = {'france', 'portugal', 'belgium', 'switzerland', 'germany', 'spain', 'norway', 'italy'}

if __name__ == '__main__':
  embedding_filename = sys.argv[1]
  K = int(sys.argv[2])
  num_dims = int(sys.argv[3]) # number of dimensions to do search over
  path = sys.argv[4] # path where to write plot
  
  vocab, embeddings = read_embedding_file(embedding_filename)
  dim = len(embeddings[0])
  random_amt = 100

  ### perform t-SNE   
  X = np.zeros(shape=(len(countries) * (1 + K) + random_amt, num_dims))
  cnt = 0
  plt_label = []
  for word in countries:
    sims, z_vals, top_k_idxs = get_nn(vocab, embeddings, word, K, num_dims)
    
    X[cnt,] = embeddings[vocab.index(word)][:num_dims]
    plt_label.append(word)
    for i, idx in enumerate(top_k_idxs):
      X[cnt+i,] = embeddings[idx][:num_dims]
      plt_label.append(vocab[idx])

    cnt += 1 + K
  
  ### add random countries
  for i in range(0, random_amt):
    X[cnt+i,] = embeddings[randint(0,len(vocab))][:num_dims]

  model = TSNE(n_components=2, random_state=0)
  Y = model.fit_transform(X)

  ### set up plot with title and label for points 
  plt.figure(figsize=(40,40))
  plt.scatter(Y[:,0], Y[:,1])
  plt.title("Countries restricted to " + str(num_dims) + " dims")
 
  for i,label in enumerate(plt_label): 
    plt.annotate(label, xy = (Y[i,0], Y[i,1]), xytext = (0, 0),
        textcoords = 'offset points', ha = 'right', va = 'bottom') 
  
  plot_name = 'countries_K=' + str(K) + '_dims=' + str(num_dims)
  plt.savefig(path + plot_name + ".jpg")
