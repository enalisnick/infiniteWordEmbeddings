import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
import cPickle as cp
from math import log, pow, exp
import sys, getopt
# Our libraries
from Evaluation.eval_lib import read_embedding_file
from Evaluation.eval_lib import get_nn
from Auto_Eval.auto_eval_iSG import process_embeddings_dir
from Evaluation.get_nearest_neighbors import get_nearest_neighbors 

# Compute p(z|w,c1,...,c_k) for input_embedding w and context_embeddings c1,...,c_k
def compute_p_z_given_w_C(input_embedding, context_embeddings, sparsity_weight=0.001, dim_penalty=1.1):
    context_size = len(context_embeddings)
    max_dim = len(context_embeddings[0]) 
    p_z_given_w_C = np.zeros(max_dim)
 
    for i in xrange(max_dim):
       context_sum = 0
       context_norm = 0
       for context_embed in context_embeddings:
           context_sum += context_embed[i]
	   context_norm += context_embed[i] * context_embed[i]

       val = -input_embedding[i]*context_sum + context_size*log(dim_penalty) + context_size*sparsity_weight*input_embedding[i]*input_embedding[i] + sparsity_weight*context_norm
       
       for j in xrange(i, max_dim):
	   p_z_given_w_C[j] += val
       
       p_z_given_w_C[i] = exp((1.0/context_size) * -p_z_given_w_C[i]) 
  
    return p_z_given_w_C / p_z_given_w_C.sum()

def get_nearest_neighbors_dot(word_embedding, in_word_idx, context_embeddings, z, k):
    word_embedding = np.array(word_embedding[:z])
    scores = np.zeros(len(context_embeddings))
    for idx, context_embedding in enumerate(context_embeddings):
        context_embedding = np.array(context_embedding[:z])
        scores[idx] = np.dot(word_embedding, np.array(context_embedding))
    if in_word_idx < len(scores): scores[in_word_idx] = -100000
    return np.argsort(-scores)[:k]

# map list of indices to list of embeddings                                     
def map2embeddings(embeddings, c_arr):                                          
  arr = []                                                                      
  for c in c_arr:                                                               
    arr.append(embeddings[c])                                                   
                                                                                
  return arr

def plot(in_vocab_all, in_embeddings_all, in_vocab, in_embeddings, out_vocab, 
    out_embeddings, sparsity, dim_penalty, words_to_plot, sentence_embeddings=[], filename=""):
    k = 20000 # truncate the vocabulary to the top k most frequent words
    num_of_modes_to_plot = 5
    num_of_nns_to_get = 3
    
    d = len(in_embeddings[0])
    # initialize subplots
    f, axarr = plt.subplots(len(words_to_plot),squeeze=False)
    #f.set_size_inches(6, 10)

    for plot_idx, word_to_plot in enumerate(words_to_plot):
        in_word_idx = in_vocab_all.index(word_to_plot)
        # compute p(z | w)
        print "computing p(z | w = %s, C)" % (word_to_plot)
        word_in_embedding = in_embeddings_all[in_word_idx]
	
        if len(sentence_embeddings)>0:
            p_z_w_C = compute_p_z_given_w_C(word_in_embedding, sentence_embeddings, sparsity, dim_penalty)
        else:
            p_z_w_C = compute_p_z_given_w_C(word_in_embedding, out_embeddings, sparsity, dim_penalty)
        print p_z_w_C
        # find nearest neighbors at the modes
        sorted_prob_idx = np.argsort(-1*p_z_w_C) # negative one so the sort is descending
        nns_at_modes = []
        modes_used = []
        idx = 0
        save_num_of_modes = num_of_modes_to_plot
        while num_of_modes_to_plot > 0:
            current_idx = sorted_prob_idx[idx]
            # check if this idx is too close to previous ones
            mode_flag = False
            if (current_idx==0 and p_z_w_C[current_idx]>p_z_w_C[current_idx+1]) or (current_idx==d-1 and p_z_w_C[current_idx]>p_z_w_C[current_idx-1]) or (p_z_w_C[current_idx]>p_z_w_C[current_idx-1] and p_z_w_C[current_idx]>p_z_w_C[current_idx+1]):
                mode_flag = True
            for mode in modes_used:
                if abs(mode[0]-current_idx) < 12:
                    mode_flag = False
            if mode_flag:
                # get nearest neighbors at current idx
                modes_used.append((current_idx, p_z_w_C[current_idx]))
                nns_at_modes.append([out_vocab[j] for j in get_nearest_neighbors_dot(word_in_embedding, in_word_idx, out_embeddings, current_idx+1, num_of_nns_to_get).tolist()])
                num_of_modes_to_plot -= 1
            idx += 1
            if idx >= d:
                break
        num_of_modes_to_plot = save_num_of_modes

        # plotting the distribution
        axarr[plot_idx,0].bar([x+1 for x in range(d)], p_z_w_C, width=1.0, facecolor='blue', edgecolor="blue")
    
        # plot the nearest neighbors at the modes
        for mode_loc, mode_nns in zip(modes_used, nns_at_modes):
            axarr[plot_idx,0].annotate(', '.join(mode_nns), xy=(mode_loc[0]+1, mode_loc[1]+0.001),  xycoords='data', xytext=(mode_loc[0]+5, mode_loc[1]+0.005), bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),arrowprops=dict(facecolor='black', shrink=0.05, frac=0.1, headwidth=2, width=1))
        axarr[plot_idx,0].set_title("p(z|w="+word_to_plot+",C)")
        axarr[plot_idx,0].set_xlim([1,d])
        axarr[plot_idx,0].set_ylim([0,modes_used[0][1]+0.009])

    # save figure
    if filename == "":
      f.savefig("p_z_given_w_C_for_"+"_".join(words_to_plot)+".png")
    else:
      f.savefig(filename+".png")


if __name__ == '__main__': 
    k = 15000 # truncate the vocabulary to the top k most frequent words
    sparsity = 0.0
    dim_penalty = 0.0
    help_message = 'graph_p_z.py -r <path to directory containing embedding files> -w <comma separated list of words to plot> -s <optional comma separated list of words to marginalize over>'

    # read in command line arguments
    input_embedding_file = ""
    context_embedding_file = ""
    words_to_plot = []
    # marginalize over given context of just use full vocab subset
    sentence_to_marginalize_over = []
   
    # filename
    filename = "" 
   
    # get args
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hr:c:w:s:n:",["rootDir=","words=","sentence=","filename="])
    except getopt.GetoptError:
        print help_message
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_message
            sys.exit()
        elif opt in ("-r", "--rootDir"):
            rootDir = arg 
        elif opt in ("-w", "--words"):
            words_to_plot = arg.split(',')
        elif opt in ("-s", "--sentence"):
            sentence_to_marginalize_over = arg.split(',')
        elif opt in ("-n", "--filename"):
            filename = arg
 
    # extract dim and sparsity penalties from file names
    arr = process_embeddings_dir(rootDir) 
    for input_embedding_file,context_embedding_file,sparsity,dim_penalty in arr:
	print 'Input embeddings file: ', input_embedding_file
	print 'Context embeddings file: ', context_embedding_file
	print 'words_to_plot: ', ", ".join(words_to_plot)
	print 'sentence to marginalize over: ', " ".join(sentence_to_marginalize_over)
	print 'sparsity penalty: ', str(sparsity)
	print 'dimension penalty: ', str(dim_penalty)


	print "loading embeddings and vocabulary..."
	in_vocab_all, in_embeddings_all = read_embedding_file(input_embedding_file)
	in_vocab = in_vocab_all[:k]
	in_embeddings = in_embeddings_all[:k]
	out_vocab_all, out_embeddings_all = read_embedding_file(context_embedding_file)
	out_vocab = out_vocab_all[:k]
	out_embeddings = out_embeddings_all[:k]
	# if a sentence is specified, get embeddings
	sentence_embeddings = []
	if len(sentence_to_marginalize_over) > 0:
	    for word in sentence_to_marginalize_over:
	       vocab_idx = -1
	       try: 
		 vocab_idx = out_vocab_all.index(word)
	       except ValueError:
		   pass  
	       sentence_embeddings.append(out_embeddings_all[vocab_idx])

	plot(in_vocab_all, in_embeddings_all, in_vocab, in_embeddings, out_vocab,
	 out_embeddings, sparsity, dim_penalty, words_to_plot, sentence_embeddings, filename)
