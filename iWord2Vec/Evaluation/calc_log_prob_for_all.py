import os
from os import listdir
from os.path import isfile, join

logspath = "/Users/enalisnick/Desktop/iSG/logs/"
vecspath = "/Users/enalisnick/Desktop/vectors/"
files = [ f for f in listdir(logspath) if isfile(join(logspath,f)) ]
roots = []
for file in files:
    roots.append(file.split('PROGRESS_')[1].split('.log')[0])
roots = set(roots)
for root in roots:
    sparsity = float(root.split('_')[1])
    dim_penalty = float(root.split('_')[2])
    print "Calculating log probability for settings: %s" %(root)
    os.system("./test_log_prob %s %s /Users/enalisnick/Desktop/word2vec/wiki_test_subset.txt %f %f" %(vecspath+'INPUT_VECS_'+root+'.txt', vecspath+'CONTEXT_VECS_'+root+'.txt', sparsity, dim_penalty))
