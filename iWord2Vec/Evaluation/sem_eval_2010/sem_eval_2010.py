import sys, re
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import numpy as np
from textwrap import wrap

from Evaluation.eval_lib import read_embedding_file
from Evaluation.scws_eval import map2embeddings, map2vocab 
from Visualization.graph_p_z import graph_p_z_w, compute_p_z_given_w
from Visualization.graph_p_z_weighted import graph_p_z_w_weighted

TEST_DIR = 'Evaluation/sem_eval_2010/test_data/nouns/'
GROUND_TRUTH_FILE = 'Evaluation/sem_eval_2010/ground_truth.txt'

'''
  Read ground truth file and return dict where key is word 
  and value is list of clusters. 
  Format of ground truth file is:
  <word.n> <word.n.example> <cluster.n.#>
def read_ground_truth(ground_truth_file):
  d = {}
  with open(ground_truth_file) as f:
    for line in f.readlines():
      word, example, cluster = line.split()
      postfix = ""
      if ".n" in arr[0]:
        postfix = ".n"
      elif ".v" in arr[0]:
        postfix = ".v" 
      word = re.search("(.*)" + postfix, word).group(1)
      print(word)
      if word not in d: d[word] = []
      l = []
      for w in arr[1:]:
        l.append(w)
      d[word].append(l)

  print(d['access'])
'''

'''
  Return dictionary mapping word to list of examples 
'''
def read_test_xml(vocab, test_xml_dir):
  onlyfiles = [ f for f in listdir(test_xml_dir) if isfile(join(test_xml_dir,f)) ]
  data = {}
 
  for f in onlyfiles:
    xml_file = test_xml_dir + f    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    word = re.search('(.*).n.test', root.tag).group(1)
    word_idx = vocab.index(word)
    print(word) 
    for child in root:
      example_name = child.tag
      sentence = "".join(child.itertext())
      sentence = sentence.replace('.','')
      context_idxs = map2vocab(vocab, sentence.replace(word, '').split()) 
      if word not in data:
        data[word] = []
      data[word].append((example_name, sentence, word_idx, context_idxs))
    
    if word == "market": break
 
  return data

if __name__ == '__main__':
  embedding_filename = sys.argv[1]
  context_embedding_filename = sys.argv[2] 
  
  print("Using embedding file: %s" % (embedding_filename))
  vocab, embeddings = read_embedding_file(embedding_filename)
  _, context_embeddings = read_embedding_file(context_embedding_filename)
  #read_ground_truth(GROUND_TRUTH_FILE)
  test_data = read_test_xml(vocab, TEST_DIR)
  print("Finished reading test data")

  for example_name, sentence, word_idx, context_idxs in test_data['market']:
    w = embeddings[word_idx]
    p_z_w = compute_p_z_given_w(w, map2embeddings(context_embeddings, context_idxs))
    print(example_name)
    #print(p_z_w)
    plt = graph_p_z_w(np.array(p_z_w), word_idx, w, embeddings, context_embeddings, vocab)
    plt.title("\n".join(wrap(sentence, 60))) 
    plt.subplots_adjust(top=0.5)
    plt.savefig("Visualization/p_z_w_"+example_name+".png")
    plt.clf()
