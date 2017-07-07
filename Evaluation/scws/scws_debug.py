import sys
import os
from Evaluation.eval_lib import read_embedding_file
from Evaluation.scws.scws_eval import map2embeddings, map2vocab, read_scws 
from Evaluation.graph_p_z import plot
SCWS_FILE = "Evaluation/scws/ratings.txt"
word1 = 'brazil'
word2 = 'nut'
context1 = ['gap', 'in', 'income', 'between', 'blacks', 'and', 'other', 'non-whites', 'is', 'relatively', 'small', 'compared', 'to', 'the', 'large', 'gap', 'between', 'whites', 'and', 'non-whites', '.', 'other', 'factors', 'such', 'as', 'illiteracy', 'and', 'education', 'level', 'show', 'the', 'same', 'patterns', '.', 'unlike', 'in', 'the', 'us', 'where', 'african', 'americans', 'were', 'united', 'in', 'the', 'civil', 'rights', 'struggle', ',', 'in', '', 'brazil', 'the', 'philosophy', 'of', 'whitening', 'has', 'helped', 'divide', 'blacks', 'from', 'other', 'non-whites', 'and', 'prevented', 'a', 'more', 'active', 'civil', 'rights', 'movement', '.', 'though', 'afro-brazilians', 'make', 'up', 'half', 'the', 'population', 'there', 'are', 'very', 'few', 'black', 'politicians', '.', 'the', 'city', 'of', 'salvador', ',', 'bahia', 'for', 'instance', 'is', '80', '%', 'afro-brazilian', 'but', 'has', 'never']
context2 = ['of', 'the', 'neck', ',', 'bridge', ',', 'and', 'pickups', ',', 'there', 'are', 'features', 'which', 'are', 'found', 'in', 'almost', 'every', 'guitar', '.', 'the', 'photo', 'below', 'shows', 'the', 'different', 'parts', 'of', 'an', 'electric', 'guitar', '.', 'the', 'headstock', '(', '1', ')', 'contains', 'the', 'metal', 'machine', 'heads', ',', 'which', 'are', 'used', 'for', 'tuning', ';', 'the', 'nut', '(', ')', ',', 'a', 'thin', 'fret-like', 'strip', 'of', 'metal', ',', 'plastic', ',', 'graphite', 'or', 'bone', 'which', 'the', 'strings', 'pass', 'over', 'as', 'they', 'first', 'go', 'onto', 'the', 'fingerboard', ';', 'the', 'machine', 'heads', '(', ')', ',', 'which', 'are', 'worm', 'gears', 'which', 'the', 'player', 'turns', 'to', 'change', 'the', 'string', 'tension']
bad_chars = ['(', ')', ',', '.', '', ' ', ';']

def map2word(vocab, idx_arr):
  arr = []
  for idx in idx_arr:
    word = ""
    try:
      word = vocab[idx]
      arr.append(word)
    except ValueError:
      pass
     
  return arr

if __name__ == '__main__':
  k = 15000 
  sparsity = 0.001
  dim_penalty = 1.1
  embedding_filename = sys.argv[1]                                              
  context_embedding_filename = sys.argv[2]                                      
  print("Using embedding file: %s" % (embedding_filename))                      
  vocab, embeddings = read_embedding_file(embedding_filename)                   
  out_vocab, context_embeddings = read_embedding_file(context_embedding_filename)
  test_data = read_scws(vocab, scws_file=SCWS_FILE, amount=1000)
  test_data = sorted(test_data, key=lambda x : x[0], reverse=True)[:10] 
  
  in_vocab = vocab[:k]
  in_embeddings = embeddings[:k]
  out_vocab = out_vocab[:k]
  out_embeddings = context_embeddings[:k]
  i = 0
  for scores,w1_idx,w2_idx,c1_idx_arr,c2_idx_arr in test_data:
    word1 = vocab[w1_idx]
    word2 = vocab[w2_idx]
    context1 = map2embeddings(context_embeddings, c1_idx_arr)
    context2 = map2embeddings(context_embeddings, c2_idx_arr) 
    context1 = [x for x in context1 if x not in bad_chars]
    context2 = [x for x in context2 if x not in bad_chars]
    #w1_idx = vocab.index(word1)
    #w2_idx = vocab.index(word2)
    #c1_idxs = map2embeddings(context_embeddings, context1)
    #c2_idxs = map2embeddings(context_embeddings, context2)
    plot(vocab, embeddings, in_vocab, in_embeddings, out_vocab, 
     out_embeddings, sparsity, dim_penalty, [word1], context1, "scws1_" + word1 + "_" + str(i))
    plot(vocab, embeddings, in_vocab, in_embeddings, out_vocab,                
     out_embeddings, sparsity, dim_penalty, [word2], context2, "scws2_" + word2 + "_" + str(i))
    plot(vocab, embeddings, in_vocab, in_embeddings, out_vocab,                
     out_embeddings, sparsity, dim_penalty, [word1], [], "p_z_w1_" + word1 + "_" + str(i))
    plot(vocab, embeddings, in_vocab, in_embeddings, out_vocab,                
     out_embeddings, sparsity, dim_penalty, [word2], [], "p_z_w2_" + word2 + "_" + str(i))
    #os.system("python Evaluation/graph_p_z.py -i %s -c %s -w %s -s %s -n %s" 
    # % (embedding_filename, context_embedding_filename, word1, ",".join(context1), "scws_" + word1))
    #os.system("python Evaluation/graph_p_z.py -i %s -c %s -w %s -s %s -n %s"
    # % (embedding_filename, context_embedding_filename, word2, ",".join(context2), "scws_" + word2))
    #os.system("python Evaluation/graph_p_z.py -i %s -c %s -w %s" 
    # % (embedding_filename, context_embedding_filename, word1)) 
    #os.system("python Evaluation/graph_p_z.py -i %s -c %s -w %s" 
    # % (embedding_filename, context_embedding_filename, word2))
    i += 1
