import sys
from Evaluation.eval_lib import read_embedding_file
from Evaluation.scws.scws_eval import map2embeddings, map2vocab
from Evaluation.sem_eval_2010.sem_eval_2010 import avg_adj_rand_index, get_clusters, clean_clusters 


WWSI_FILE = 'Evaluation/wwsi/wwsi.txt' 

# Return ground truth and
# examples in form of list of (word_idx, context_idxs) 
def read_wwsi_file(vocab, wwsi_file=WWSI_FILE):
  test_data = {}
  ground_truth_clusters_dict  = {}
  #word_cnt = 0
  with open(wwsi_file) as f:
    line = f.readline()
    while line:
      # Read word and number of meanings 
      _,_,word,_,num_meanings = line.split() 
      try:                                                                      
        word_idx = vocab.index(word)                                            
      except ValueError:                                                        
        pass
      print word, num_meanings
      test_data[word] = []
      ground_truth_clusters_dict[word] = []
      
      # Each each meaning 
      ground_truth_clusters_dict[word] = {}
      cnt = 0
      for i in range(0,int(num_meanings)):
	num_sentences = f.readline().split()[-1]
	#print num_sentences
	cluster = []
	# Read all sentence for this meaning
	for j in range(0,int(num_sentences)):
	  word_pos = f.readline()
	  contexts = f.readline().split() 
	  context_idxs = map2vocab(vocab, contexts)
	  example = word + '_' + str(i) + '.' + str(cnt)
          test_data[word].append((example, word_idx, context_idxs))
	  cluster.append(example)
	  cnt += 1

	ground_truth_clusters_dict[word][i] = cluster
      line = f.readline()
      #word_cnt += 1
      #if word_cnt > 2: break
  return ground_truth_clusters_dict, test_data
 
if __name__ == '__main__':
  # read embedding files                                                        
  embedding_filename = sys.argv[1]                                              
  context_embedding_filename = sys.argv[2]                                      
  print("Using embedding file: %s" % (embedding_filename))                      
  vocab, embeddings = read_embedding_file(embedding_filename)                   
  _, context_embeddings = read_embedding_file(context_embedding_filename)
 
  ground_truth_clusters_dict, test_data = read_wwsi_file(vocab)
  print("read wwsi file")
  iw2v_clusters_dict = clean_clusters(get_clusters(test_data, vocab, embeddings, context_embeddings))
  rand_index = avg_adj_rand_index(ground_truth_clusters_dict, iw2v_clusters_dict)
  print 'avg_rand_index: ', rand_index 
