import os
import subprocess
from os import listdir
from os.path import isfile, join

logspath = "/Users/enalisnick/Desktop/iSG/logs/"
vecspath = "/Users/enalisnick/Desktop/vectors/"
files = [ f for f in listdir(logspath) if isfile(join(logspath,f)) ]
roots = []
for file in files:
    if '0.05.log' in file: # only get the ones with higher learning rates
        roots.append(file.split('PROGRESS_')[1].split('.log')[0])
roots = set(roots)
for root in roots:
    sparsity = float(root.split('_')[1])
    dim_penalty = float(root.split('_')[2])
    call_args = '%s %s /Users/enalisnick/Desktop/word2vec/wiki_tiny_subset.txt /Users/enalisnick/Desktop/wiki_1B_vocab.txt %.8f %.8f' %(vecspath+'INPUT_VECS_'+root+'.txt', vecspath+'CONTEXT_VECS_'+root+'.txt', sparsity, dim_penalty)
    print "Calculating log probability for settings: %s" %(root)
    p = subprocess.check_output(['/Users/enalisnick/GitHub/infiniteWordEmbeddings/iWord2Vec/Evaluation/test_log_prob', call_args])
    print p
    exit()
