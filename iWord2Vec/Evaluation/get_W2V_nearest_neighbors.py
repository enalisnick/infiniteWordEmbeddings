import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
import cPickle as cp
from math import log, pow, exp
import sys, getopt
from scipy import spatial
from tabulate import tabulate
# Our libraries
from Evaluation.eval_lib import read_embedding_file
from Evaluation.eval_lib import get_nn

def get_nearest_neighbors(word_embedding, in_word_idx, context_embeddings, k):
    scores = np.zeros(len(context_embeddings))
    for idx, context_embedding in enumerate(context_embeddings):
        scores[idx] = 1 - spatial.distance.cosine(context_embedding, word_embedding)
    scores[in_word_idx] = -100000
    return np.argsort(-scores)[:k]

if __name__ == '__main__':
    # hard coded parameters
    k = 5000 # truncate the vocabulary to the top k most frequent words
    num_of_nns_to_get = 5
    headers = ["Rank", "Word"]
    help_message = 'get_nearest_neighbors.py -i <input embeddings file> -w <comma separated list of words to plot> -k <number of neighbors to get>'

    # read in command line arguments
    input_embedding_file = ""
    context_embedding_file = ""
    words_to_plot = []
    # get args
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:w:k:",["ifile=","cfile=","words=","numNeighbors","sentence="])
    except getopt.GetoptError:
        print help_message
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_message
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_embedding_file = arg
        elif opt in ("-w", "--words"):
            words_to_plot = arg.split(',')
        elif opt in ("-k", "--numNeighbors"):
            num_of_nns_to_get = int(arg)

    print 'Input embeddings file: ', input_embedding_file
    print 'words_to_plot: ', ", ".join(words_to_plot)
    
    print "loading embeddings and vocabulary..."
    in_vocab, in_embeddings = read_embedding_file(input_embedding_file)
    in_vocab = in_vocab[:k]
    in_embeddings = in_embeddings[:k]
    d = len(in_embeddings[0])
    
    for plot_idx, word_to_plot in enumerate(words_to_plot):

        in_word_idx = in_vocab.index(word_to_plot)
        word_in_embedding = in_embeddings[in_word_idx]
        
        nn_idxs = get_nearest_neighbors(word_in_embedding, in_word_idx, in_embeddings, num_of_nns_to_get)
        
        t = []
        for i, idx in enumerate(nn_idxs):
            t.append([i+1, in_vocab[idx]])
        # output table                                                                                                                                                                                            
        print tabulate(t,headers=headers)
        print
        print


