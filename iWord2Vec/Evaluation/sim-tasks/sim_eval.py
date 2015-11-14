import sys
import numpy as np
from scipy import spatial
from Evaluation.eval_lib import read_embedding_file, dot_prod_sim, cosine_sim, get_rank_corr_for_sim 

SIM_FILE = {}
SIM_FILE[0] = 'Evaluation/sim-tasks/wordSim353_sorted.csv'
SIM_FILE[1] = 'Evaluation/sim-tasks/MEN_sorted.txt'

if __name__ == '__main__':
  ### READ INPUT ARGS
  use_full_dim = 0
  num_of_args = len(sys.argv)
  sim_filename = SIM_FILE[0]
  if num_of_args < 2:
    print "Embedding file not specified...quitting."
    exit()

  embedding_filename = sys.argv[1] 
  context_embedding_filename = sys.argv[2]
  if num_of_args > 3:
    sim_filename = SIM_FILE[int(sys.argv[3])]

  if num_of_args > 4:
    use_full_dim = int(sys.argv[4])
  
  print("Using embedding file: %s" % (embedding_filename))
  print("Using similarity benchmark: %s" % (sim_filename))
  vocab, embeddings = read_embedding_file(embedding_filename) 
  _, context_embeddings = read_embedding_file(context_embedding_filename)

  corr = 0 
  if use_full_dim == 0:
    print("Using prob derived dims")
    corr = get_rank_corr_for_sim(sim_filename, vocab, embeddings, context_embeddings, False)
  else:
    print("Using all dims")
    corr = get_rank_corr_for_sim(sim_filename, vocab, embeddings, True)

  print(corr) 
