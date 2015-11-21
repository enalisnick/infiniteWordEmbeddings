import sys, re
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import numpy as np
from textwrap import wrap
import math

from Evaluation.eval_lib import read_embedding_file
from Evaluation.scws.scws_eval import map2embeddings, map2vocab 
from Evaluation.graph_p_z import compute_p_z_given_w

TEST_DIR = 'Evaluation/sem_eval_2010/test_data/nouns/'
GROUND_TRUTH_FILE = 'Evaluation/sem_eval_2010/ground_truth.txt'

BUCKET_SIZE = 10

# Return dictionary mapping word to list of examples 
def read_test_xml(vocab, test_xml_dir):
  onlyfiles = [ f for f in listdir(test_xml_dir) if isfile(join(test_xml_dir,f)) ]
  test_data = {}
 
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
      if word not in test_data:
        test_data[word] = []
      test_data[word].append((example_name, word_idx, context_idxs))
    
    if word == "market": break
 
  return test_data


'''  
  Read ground truth file and return dict where key is word 
  and value is dict where key is cluster name and value is list of examples
  belonging to that cluster. 
  Format of ground truth file is:
  <word.n> <word.n.example> <cluster.n.#>
'''
def read_ground_truth(ground_truth_file=GROUND_TRUTH_FILE):
  d = {}
  with open(ground_truth_file) as f:
    for line in f.readlines():
      word, example, cluster = line.split()
      postfix = ""
      if ".n" in word:
        postfix = ".n"
      elif ".v" in word:
        postfix = ".v" 
      word = re.search("(.*)"+postfix, word).group(1)
      #print word, example, cluster
      if word not in d: d[word] = {}
      if cluster not in d[word]: d[word][cluster] = []
      d[word][cluster].append(example)

  return d

def nCr(n,r):
  if r > n: return 0

  f = math.factorial
  return f(n) / f(r) / f(n-r)

# return length of intersection between two lists
def intersection_len(list1, list2):
  return len([item for item in list1 if item in list2])

# adjusted rand index from 
# https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
def adj_rand_index(clusters1, clusters2):
  #print(clusters1)
  #print '----------'
  #print(clusters2)
  # ensure both clusters have same number of items
  num_items1 = 0; num_items2 = 0
  for k1 in clusters1:
    num_items1 += len(clusters1[k1])
  for k2 in clusters2:
    num_items2 += len(clusters2[k2])
  #print 'num_items1', num_items1
  #print 'num_items2', num_items2
  assert num_items1 == num_items2

  matrix = np.zeros(shape=(len(clusters1.keys()), len(clusters2.keys())))
  for idx1, k1 in enumerate(clusters1):
    for idx2, k2 in enumerate(clusters2):
      matrix[idx1][idx2] = intersection_len(clusters1[k1], clusters2[k2])    

  numer_first_term = 0.0; numer_second_term = 0.0
  denom_first_term = 0.0; denom_second_term = 0.0

  for x in np.nditer(matrix):
    numer_first_term += nCr(x,2)

  row_sum = 0.0; col_sum = 0.0
  row_sum_arr = np.sum(matrix, axis=0)
  col_sum_arr = np.sum(matrix, axis=1)
  for x in np.nditer(row_sum_arr):
    row_sum += nCr(x,2)
  for x in np.nditer(col_sum_arr):
    col_sum += nCr(x,2)
    
  numer_second_term = (row_sum * col_sum)/nCr(num_items1,2)  
  denom_first_term = 0.5 * (row_sum + col_sum)
  denom_second_term = numer_second_term

  return ((numer_first_term - numer_second_term)/(denom_first_term - denom_second_term))

# average adjusted rand index across all words
def avg_adj_rand_index(clusters_dict1, clusters_dict2):
  #print clusters_dict1.keys()
  #print clusters_dict2.keys()
  assert len(clusters_dict1.keys()) == len(clusters_dict2.keys())
  total = 0.0
  for word in clusters_dict1:
    val = adj_rand_index(clusters_dict1[word], clusters_dict2[word])
    print word, val 
    total += val 

  return (total/len(clusters_dict1.keys()))

# remove empty clusters for every word
def clean_clusters(clusters_list):
  clusters_dict = {}
  for word in clusters_list:
    # remove empty lists from list for word and convert to dict
    clusters = clusters_list[word]
    clusters = dict(enumerate(clusters))
    clusters = {k : clusters[k] for k in clusters if clusters[k] != []}     
    clusters_dict[word] = clusters
 
  return clusters_dict

def get_mode_p_z_given_w(w, context_embeddings): 
  return np.argmax(compute_p_z_given_w(w, context_embeddings), axis=0)

def get_expectation(w, context_embeddings):
  p_z_given_w = compute_p_z_given_w(w, context_embeddings)                                                         
  exp = 0.                                                                      
  for i in range(0,len(p_z_given_w)):                                                 
    exp += (i+1) * p_z_given_w[i]                                                     
                                                                                
  return int(exp)

def get_clusters(test_data, vocab, embeddings, context_embeddings):
  iw2v_clusters = {}
  for word in test_data: 
    for example_name, word_idx, context_idxs in test_data[word]:
      word = vocab[word_idx]
      w = embeddings[word_idx]
      #mode_z = get_mode_p_z_given_w(w, map2embeddings(context_embeddings, context_idxs))
      exp_z = get_expectation(w, map2embeddings(context_embeddings, context_idxs))
      print(example_name)
      print(exp_z)
      if word not in iw2v_clusters:
	iw2v_clusters[word] = [[] for i in range(0, (len(w)/BUCKET_SIZE) + 1)]
	
      iw2v_clusters[word][exp_z/BUCKET_SIZE].append(example_name)

  return iw2v_clusters

if __name__ == '__main__':
  # get ground truth
  ground_truth_clusters_dict = read_ground_truth()

  # read embedding files 
  embedding_filename = sys.argv[1]
  context_embedding_filename = sys.argv[2]
  print("Using embedding file: %s" % (embedding_filename))
  vocab, embeddings = read_embedding_file(embedding_filename)
  _, context_embeddings = read_embedding_file(context_embedding_filename)
  
  # read test data 
  test_data = read_test_xml(vocab, TEST_DIR)
  print("Finished reading test data")

  iw2v_clusters_dict = clean_clusters(get_clusters(test_data, vocab, embeddings, context_embeddings)) 
  '''
  plt = graph_p_z_w(np.array(p_z_w), word_idx, w, embeddings, context_embeddings, vocab)
  plt.title("\n".join(wrap(sentence, 60))) 
  plt.subplots_adjust(top=0.5)
  plt.savefig("Visualization/p_z_w_"+example_name+".png")
  plt.clf()
  '''
  market_dict = {}
  market_dict['market'] = ground_truth_clusters_dict['market']
  print avg_adj_rand_index(market_dict, iw2v_clusters_dict) 
