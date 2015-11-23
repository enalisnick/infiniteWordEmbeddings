import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
from math import log, pow, exp
import getopt

### Get vocab and word-embeddings from file 
def read_embedding_file(embedding_filename):
  ### READ EMBEDDINGS FROM TXT FILE
  embeddings = []
  vocab = []
  with open(embedding_filename) as f:
    for line in f.readlines()[1:]:
      line = line.strip().split()
      vocab.append(line[0])
      embeddings.append([float(x) for x in line[1:]])
  return vocab, embeddings  

### Calculate p(z|w) ###
def compute_unnorm_z_probs_recursively(in_vec, out_vec, max_dim, sparsity_weight, dim_penalty):
    z_probs = np.zeros(max_dim)
    for idx1 in xrange(max_dim):
        val = -in_vec[idx1]*out_vec[idx1] + log(dim_penalty) + sparsity_weight*pow(in_vec[idx1],2) + sparsity_weight*pow(out_vec[idx1],2)
        for idx2 in xrange(idx1, max_dim):
            z_probs[idx2] += val
        z_probs[idx1] = exp(-z_probs[idx1])
    return z_probs

def compute_p_z_given_w(input_embedding, context_embedding, sparsity_weight=0.001, dim_penalty=1.1):
    d = len(context_embedding)
    p_z_given_w = np.zeros(d)
    p_z_given_w += compute_unnorm_z_probs_recursively(input_embedding, context_embedding, d, sparsity_weight, dim_penalty)
    return p_z_given_w / p_z_given_w.sum()

### Sim metrics ###
def cosine_sim(v1,v2):
  return (1.0 - spatial.distance.cosine(v1,v2))

def inner_prod(v1,v2):
  return np.dot(v1,v2)

### Calculate Spearman
def compute_spearman_rank(human_sims, model_sims):
  ### Human sims have been sorted, model sims have not been
  model_ranks = [i[0] for i in sorted(enumerate(model_sims), key=lambda x:x[1])]
  n = len(human_sims)
  sum_squ_distances = 0.
  for idx, val in enumerate(model_ranks):
    sum_squ_distances += (idx-val)**2
  return (1.0 - (6*sum_squ_distances)/(n*(n**2 - 1.0)))

### Get sim task performance ###
def perform_word_sim_task(vocab, input_embeddings, context_embeddings, sparsity, dim_penalty, sim_file, outF):
  model_sims = []
  human_sims = []
  w1_embeddings = []
  w2_embeddings = []
  metrics = [inner_prod] #cosine_sim]
  ### read in task sims and needed embeddings
  with open(sim_file) as f:
    for line in f:
      if ',' in line:
        line = line.strip().split(',')  # for sim353
      else:
        line = line.strip().split() # for MEN
      try:
        word1_idx = vocab.index(line[0].lower())
        word2_idx = vocab.index(line[1].lower())
        human_sims.append(float(line[2]))
        w1_embeddings.append(input_embeddings[word1_idx])
        w2_embeddings.append(context_embeddings[word2_idx])
      except ValueError:
        continue

  for weighted_flag in [1]: # ,0]:
    for metric in metrics:
      outF.write("Using metric: %s,    "%(metric.__name__))
      outF.write("Weighted by p(z|w1,w2)?: %s \n"%(bool(weighted_flag)))
      for v1, v2 in zip(w1_embeddings, w2_embeddings):
        if weighted_flag == 1:
          p_z_w1_w2 = compute_p_z_given_w(v1, v2, sparsity, dim_penalty)
          temp_sum = 0.0
          for idx, z_weight in enumerate(p_z_w1_w2): 
            temp_sum += z_weight * metric(v1[:idx+1],v2[:idx+1])
          model_sims.append(temp_sum)
        else:
          model_sims.append(metric(v1,v2))
      outF.write("Spearman's Rank Correlation: %.4f \n\n" %(compute_spearman_rank(human_sims, model_sims)))
      model_sims = []
    
##### MAIN #####
if __name__ == '__main__': 
    k = 15000 # truncate the vocabulary to the top k most frequent words
    sparsity = 0.0
    dim_penalty = 0.0
    sim_files = {}
    sim_files['wordsim353'] = '../Evaluation/sim-tasks/wordSim353_sorted.csv'
    sim_files['MEN'] = '../Evaluation/sim-tasks/MEN_sorted.txt'
    rootDir = ""
    outputFile = ""
    help_message = 'auto_eval_iSG.py -r <path to directory containing embedding files> -o <name of output file>'
    # get args                                                                                                                                                                   
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hr:o:",["rootDir=","outputFile="])
    except getopt.GetoptError:
        print help_message
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_message
            sys.exit()
        elif opt in ("-r", "--rootDir"):
            rootDir = arg
        elif opt in ("-o", "--outputFile"):
            outputFile = arg

    outF = open("iSG_WordSim_results.out","w")
    outF.write("Evaluating embedding files in directory: %s \n\n" %(rootDir))

    # get the various parameter settings
    files = [ f for f in listdir(rootDir) if isfile(join(rootDir,f)) ]
    roots = []
    for file in files:
        if '0.05.txt' in file: # add contraints here if only want to eval a subset of files 
            roots.append(file.split('_VECS_')[1].split('.txt')[0])
    roots = set(roots)
    for root in roots:
        sparsity = float(root.split('_')[1])
        dim_penalty = float(root.split('_')[2])
    
        input_embedding_file = "INPUT_VECS_"+root+".txt"
        context_embedding_file = "CONTEXT_VECS_"+root+".txt"

        outF.write('#####################################################\n')
        outF.write('Input embeddings file: %s \n' %(input_embedding_file))
        outF.write('Context embeddings file: %s \n' %(context_embedding_file))
        outF.write('Sparsity penalty: %.8f \n' %(sparsity))
        outF.write('Dimension penalty: %.2f \n\n' %(dim_penalty))

        vocab, in_embeddings = read_embedding_file(rootDir+input_embedding_file)
        #in_vocab = in_vocab[:k]
        #in_embeddings = in_embeddings[:k]
        _, out_embeddings = read_embedding_file(rootDir+context_embedding_file)
        #out_vocab = out_vocab[:k]
        #out_embeddings = out_embeddings[:k]

        for sim_task in sim_files.keys():
            outF.write("Evaluating on %s task\n" %(sim_task))
            perform_word_sim_task(vocab, in_embeddings, out_embeddings, sparsity, dim_penalty, sim_files[sim_task], outF)
            outF.write("\n\n")
    outF.close()
