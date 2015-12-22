import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
import cPickle as cp
from math import log, pow, exp, sqrt
import sys, getopt
from tabulate import tabulate
# Our libraries
from Evaluation.eval_lib import read_embedding_file
from Evaluation.eval_lib import get_nn

def compute_unnorm_z_probs_recursively(in_vec, out_vec, max_dim, sparsity_weight, dim_penalty):
    z_probs = np.zeros(max_dim)
    for idx1 in xrange(max_dim):
        val = -in_vec[idx1]*out_vec[idx1] + log(dim_penalty) + sparsity_weight*pow(in_vec[idx1],2) + sparsity_weight*pow(out_vec[idx1],2)
        for idx2 in xrange(idx1, max_dim):
            z_probs[idx2] += val
        z_probs[idx1] = exp(-z_probs[idx1])
    return z_probs

def compute_p_z_given_w(input_embedding, context_embeddings, sparsity_weight=0.001, dim_penalty=1.1):
    n = len(context_embeddings)
    d = len(context_embeddings[0])
    p_z_given_w = np.zeros(d)
    for context_vec in context_embeddings:
        p_z_given_w += compute_unnorm_z_probs_recursively(input_embedding, context_vec, d, sparsity_weight, dim_penalty)
    return p_z_given_w / p_z_given_w.sum()

def compute_p_z_given_w_c(input_embedding, context_embeddings, sparsity_weight=0.001, dim_penalty=1.1):
    n = len(context_embeddings)
    d = len(context_embeddings[0])
    p_z_given_w_c = np.zeros(n,d)
    for idx,context_vec in enumerate(context_embeddings):
        p_z_given_w_c[idx,:] = compute_unnorm_z_probs_recursively(input_embedding, context_vec, d, sparsity_weight, dim_penalty)
    return p_z_given_w_c / p_z_given_w.sum(axis=1)

def get_nearest_neighbors(word_embedding, in_word_idx, context_embeddings, p_z_given_w_c, k):
    scores = np.zeros(len(context_embeddings))
    for idx, context_embedding in enumerate(context_embeddings):
        temp_sum = 0.0
        for dim_idx in xrange(len(p_z_given_w)):
            temp_sum += word_embedding[dim_idx]*context_embedding[dim_idx]
            scores[idx] += p_z_given_w_c[idx,dim_idx] * temp_sum 
    scores[in_word_idx] = -100000
    return np.argsort(-scores)[:k]

if __name__ == '__main__':
    # hard coded parameters
    k = 10000 # truncate the vocabulary to the top k most frequent words
    num_of_nns_to_get = 20
    sparsity = 0.0
    dim_penalty = 0.0
    headers = ["Rank", "Word"]
    help_message = 'get_nearest_neighbors.py -i <input embeddings file> -c <context embeddings file> -w <comma separated list of words to plot> -k <number of neighbors to get> -s <optional comma separated list of words to marginalize over>'

    # read in command line arguments
    input_embedding_file = ""
    context_embedding_file = ""
    words_to_plot = []
    # marginalize over given context of just use full vocab subset
    sentence_to_marginalize_over = []
    # get args
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:c:w:k:s:",["ifile=","cfile=","words=","numNeighbors","sentence="])
    except getopt.GetoptError:
        print help_message
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_message
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_embedding_file = arg
        elif opt in ("-c", "--cfile"):
            context_embedding_file = arg
        elif opt in ("-w", "--words"):
            words_to_plot = arg.split(',')
        elif opt in ("-k", "--numNeighbors"):
            num_of_nns_to_get = int(arg)
        elif opt in ("-s", "--sentence"):
            sentence_to_marginalize_over = arg.split(',')

    # extract dim and sparsity penalties from file names
    sparsity = float(input_embedding_file.split('_')[3])
    dim_penalty = float(input_embedding_file.split('_')[4])

    print 'Input embeddings file: ', input_embedding_file
    print 'Context embeddings file: ', context_embedding_file
    print 'words_to_plot: ', ", ".join(words_to_plot)
    print 'sentence to marginalize over: ', " ".join(sentence_to_marginalize_over)
    print 'sparsity penalty: ', str(sparsity)
    print 'dimension penalty: ', str(dim_penalty)


    print "loading embeddings and vocabulary..."
    in_vocab, in_embeddings = read_embedding_file(input_embedding_file)
    in_vocab = in_vocab[:k]
    in_embeddings = in_embeddings[:k]
    out_vocab, out_embeddings = read_embedding_file(context_embedding_file)
    out_vocab = out_vocab[:k]
    out_embeddings = out_embeddings[:k]
    d = len(in_embeddings[0])
    # if a sentence is specified, get embeddings
    sentence_embeddings = []
    if len(sentence_to_marginalize_over) > 0:
        for word in sentence_to_marginalize_over:
            sentence_embeddings.append(out_embeddings[out_vocab.index(word)])

    for plot_idx, word_to_plot in enumerate(words_to_plot):

        in_word_idx = in_vocab.index(word_to_plot)
        # compute p(z | w)
        print "computing p(z | w = %s )" %(word_to_plot)
        word_in_embedding = in_embeddings[in_word_idx]
        
        if len(sentence_to_marginalize_over)>0:
            p_z_w_c = compute_p_z_given_w_c(word_in_embedding, sentence_embeddings, sparsity, dim_penalty)
        else:
            p_z_w_c = compute_p_z_given_w_c(word_in_embedding, out_embeddings, sparsity, dim_penalty)

        nn_idxs = get_nearest_neighbors(word_in_embedding, in_word_idx, out_embeddings, p_z_w_c, num_of_nns_to_get)
        
        t = []
        for i, idx in enumerate(nn_idxs):
            t.append([i+1, out_vocab[idx]])
        # output table                                                                                                                                                                                            
        print tabulate(t,headers=headers)
        print
        print


