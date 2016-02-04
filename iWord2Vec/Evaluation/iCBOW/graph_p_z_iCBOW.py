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

def compute_unnorm_z_probs_recursively(in_vec, context_embeddings, max_dim, sparsity_weight, dim_penalty):
    z_probs = np.zeros(max_dim)
    for idx1 in xrange(max_dim):
        avg_c = 0.0
        avg_c_norm = 0.0
        for c in context_embeddings:
            avg_c += c[idx1]
            avg_c_norm += c[idx1]*c[idx1]
        avg_c /= len(context_embeddings)
        avg_c_norm /= len(context_embeddings)
        val = -in_vec[idx1]*avg_c + log(dim_penalty) + sparsity_weight*in_vec[idx1]*in_vec[idx1] + sparsity_weight*avg_c_norm
        for idx2 in xrange(idx1, max_dim):
            z_probs[idx2] += val
        z_probs[idx1] = exp(-z_probs[idx1])
    return z_probs

def compute_p_z_given_w(input_embeddings, context_embeddings, sparsity_weight=0.001, dim_penalty=1.1):
    n = len(context_embeddings)
    d = len(context_embeddings[0])
    p_z_given_C = np.zeros(d)
    for w_vec in input_embeddings:
        p_z_given_C += compute_unnorm_z_probs_recursively(w_vec, context_embeddings, d, sparsity_weight, dim_penalty)
    return p_z_given_C / p_z_given_C.sum()

def get_nearest_neighbors(word_embedding, in_word_idx, context_embeddings, z, k):
    word_embedding = np.array(word_embedding[:z])
    scores = np.zeros(len(context_embeddings))
    for idx, context_embedding in enumerate(context_embeddings):
        context_embedding = np.array(context_embedding[:z])
        scores[idx] = np.dot(word_embedding, np.array(context_embedding))
    if in_word_idx < len(scores): scores[in_word_idx] = -100000
    return np.argsort(-scores)[:k]

def plot(w_vocab, w_embeddings, context_avg_vecs, context_all_vecs, sparsity, dim_penalty, context_to_plot, filename="dummy"):
    num_of_modes_to_plot = 5
    num_of_nns_to_get = 3
    
    d = len(in_embeddings[0])
    # initialize subplots
    f, axarr = plt.subplots(len(words_to_plot),squeeze=False)
    #f.set_size_inches(6, 10)

    for idx, context in enumerate(context_to_plot):
        # compute p(z | w)
        print "computing p(z | w = %s )" %(" ".join(context))
	
        p_z_w = compute_p_z_given_w(w_embeddings, context_all_vecs[idx], sparsity, dim_penalty)
  
        # find nearest neighbors at the modes
        sorted_prob_idx = np.argsort(-1*p_z_C) # negative one so the sort is descending
        nns_at_modes = []
        modes_used = []
        idx = 0
        save_num_of_modes = num_of_modes_to_plot
        while num_of_modes_to_plot > 0:
            current_idx = sorted_prob_idx[idx]
            # check if this idx is too close to previous ones
            mode_flag = False
            if (current_idx==0 and p_z_C[current_idx]>p_z_w[current_idx+1]) or (current_idx==d-1 and p_z_C[current_idx]>p_z_w[current_idx-1]) or (p_z_C[current_idx]>p_z_w[current_idx-1] and p_z_C[current_idx]>p_z_C[current_idx+1]):
                mode_flag = True
            for mode in modes_used:
                if abs(mode[0]-current_idx) < 12:
                    mode_flag = False
            if mode_flag:
                # get nearest neighbors at current idx
                modes_used.append((current_idx, p_z_C[current_idx]))
                nns_at_modes.append([out_vocab[j] for j in get_nearest_neighbors(w_embeddings, w_vocab, context_avg_vecs[idx], current_idx+1, num_of_nns_to_get).tolist()])
                num_of_modes_to_plot -= 1
            idx += 1
            if idx >= d:
                break
        num_of_modes_to_plot = save_num_of_modes

        # plotting the distribution
        axarr[plot_idx,0].bar([x+1 for x in range(d)], p_z_C, width=1.0, facecolor='black', alpha=0.7, edgecolor="black")
    
        # plot the nearest neighbors at the modes
        for mode_loc, mode_nns in zip(modes_used, nns_at_modes):
            axarr[plot_idx,0].annotate(', '.join(mode_nns), xy=(mode_loc[0]+1, mode_loc[1]+0.001),  xycoords='data', xytext=(mode_loc[0]+5, mode_loc[1]+0.005), bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.05),arrowprops=dict(facecolor='black', shrink=0.05, frac=0.1, headwidth=2, width=1))
        axarr[plot_idx,0].set_title("p(z|C="+word_to_plot+")")
        axarr[plot_idx,0].set_xlim([1,d])
        axarr[plot_idx,0].set_ylim([0,modes_used[0][1]+0.009])

    # save figure
    if filename == "":
      f.savefig("p_z_for_C.png")
    else:
      f.savefig(filename+".png")


if __name__ == '__main__': 
    k = 15000 # truncate the vocabulary to the top k most frequent words
    sparsity = 0.0
    dim_penalty = 0.0
    help_message = 'graph_p_z_iCBOW.py -r <path to directory containing embedding files> -c <comma separated list of (context) words to plot. multiple contexts separated by periods>'

    # read in command line arguments
    input_embedding_file = ""
    context_embedding_file = ""
    context_to_plot = []
   
    # filename
    filename = "" 
   
    # get args
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hr:c:",["rootDir=","words="])
    except getopt.GetoptError:
        print help_message
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print help_message
            sys.exit()
        elif opt in ("-r", "--rootDir"):
            rootDir = arg 
        elif opt in ("-c", "--context"):
            for a in arg.split('.'):
                context_to_plot.append(a.split(','))
 
    # extract dim and sparsity penalties from file names
    arr = process_embeddings_dir(rootDir) 
    for input_embedding_file,context_embedding_file,sparsity,dim_penalty in arr:
	print 'Context embeddings file: ', context_embedding_file
	print 'Input embeddings file: ', input_embedding_file
        print 'sparsity penalty: ', str(sparsity)
        print 'dimension penalty: ', str(dim_penalty)
        for i,c in enumerate(context_to_plot):
            print str(i)+': context to plot: ', " ".join(c)

	print "loading embeddings and vocabulary..."
	w_vocab, w_embeddings = read_embedding_file(input_embedding_file)
        # reduce words to sum over
	w_vocab = w_vocab[:k]
	w_embeddings = w_embeddings[:k]
	c_vocab, c_embeddings = read_embedding_file(context_embedding_file)
	# if a sentence is specified, get embeddings
	context_avg_vecs = [] 
        context_all_vecs = []
	for idx_s, s in context_to_plot:
            word_counter = 0
	    for idx_w, word in enumerate(s):
                if idx_w == 0:
                    context_avg_vecs.append(np.zeros(len(c_embeddings[0])))
                    context_all_vecs.append([])
                vocab_idx = -1
                try: 
                    vocab_idx = c_vocab.index(word)
                except ValueError:
                    continue
                context_avg_vecs[-1] += np.float(c_embeddings[vocab_idx])
                context_all_vecs[-1].append(c_embeddings[vocab_idx])
                word_counter += 1.0
            if word_counter > 0:
                context_avg_vecs[-1] *= 1.0/word_counter
            

        plot(w_vocab, w_embeddings, context_avg_vecs, context_all_vecs, sparsity, dim_penalty, context_to_plot)
