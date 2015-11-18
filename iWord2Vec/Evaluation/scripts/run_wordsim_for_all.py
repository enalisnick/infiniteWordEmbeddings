import os
from os import listdir
from os.path import isfile, join
from Evaluation.eval_lib import read_embedding_file, get_rank_corr_for_sim

SIM_FILE = {}                                                                   
SIM_FILE[0] = 'Evaluation/sim-tasks/wordSim353_sorted.csv'                      
SIM_FILE[1] = 'Evaluation/sim-tasks/MEN_sorted.txt'

logspath = "/Users/enalisnick/Desktop/iSG/logs/"
vecspath = "/Users/enalisnick/Desktop/vectors/"

'''
files = [ f for f in listdir(logspath) if isfile(join(vecspath,f)) ] 
roots = []
for file in files:
    roots.append(file.split('PROGRESS_')[1].split('.log')[0])
roots = set(roots)
'''

use_mode_z = [False, True]
use_input_to_context = [False, True]
for root in roots:
  sparsity = float(root.split('_')[1])
  dim_penalty = float(root.split('_')[2])
  embedding_filename = vecspath+'INPUT_VECS_'+root+'.txt' 
  context_embedding_filename = vecspath+'CONTEXT_VECS_'+root+'.txt'

  vocab, embeddings = read_embedding_file(embedding_filename)                   
  _, context_embeddings = read_embedding_file(context_embedding_filename) 

  print "--------------------------------------------------"
  print "MODEL: %s" % (embedding_filename)
  for i in range(0,2): # iterate sim353, MEN
    for j in range(0,2): # iterate use_mode_z
      for k in range(0,2): # iterate use_input_to_context
	 rank_sim = get_rank_corr_for_sim(SIM_FILE[i], vocab, embeddings, context_embeddings, 
	  sparsity, dim_penalty, False, use_mode_z[j], use_input_to_context[k])
	 print "SIM=%s, use_mode_z=%r, use_input_to_context=%r: %f" % (SIM_FILE[i], 
	  use_mode_z[j], use_input_to_context[k], rank_sim)

  print "--------------------------------------------------" 
