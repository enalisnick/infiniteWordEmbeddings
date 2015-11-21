import os
import subprocess
from os import listdir
from os.path import isfile, join

vecspath = "/Users/enalisnick/Desktop/word2vec/"
files = [ f for f in listdir(vecspath) if isfile(join(vecspath,f)) ]
roots = []
out_file = open("W2V_log_likelihood_results.out","w")
for file in files:
    if 'w2v_1B_100' in file: 
        roots.append(file.split('_')[-1].split('dim')[0])
roots = set(roots)
for root in roots:
    call_args = '%s %s /Users/enalisnick/Desktop/word2vec/wiki_test200MB.txt /Users/enalisnick/Desktop/wiki_1B_vocab.txt' %(vecspath+'INPUT_w2v_1B_'+root+'dim.txt', vecspath+'CONTEXT_w2v_1B_'+root+'dim.txt')
    print "Calculating log probability for dimensions: %s" %(root)
    p = subprocess.Popen(['/Users/enalisnick/GitHub/infiniteWordEmbeddings/iWord2Vec/Evaluation/W2V_test_log_prob']+call_args.split(), stdout=subprocess.PIPE)
    p.wait()
    output = p.stdout.read()
    out_file.write(output)
    out_file.write("\n")
    out_file.write("\n")
    out_file.close()
    exit()
out_file.close()
