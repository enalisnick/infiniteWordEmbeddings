import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
import cPickle as cp
from math import log, pow, exp
# Our libraries
from Evaluation.eval_lib import read_embedding_file
from Evaluation.eval_lib import get_nn

def compute_unnorm_z_probs_recursively(in_vec, out_vec, max_dim):
    sparsity_weight = 0.001
    dim_penalty = 1.1
    z_probs = np.zeros(max_dim)
    for idx1 in xrange(max_dim):
        val = -in_vec[idx1]*out_vec[idx1] + log(dim_penalty) + sparsity_weight*pow(in_vec[idx1],2) + sparsity_weight*pow(out_vec[idx1],2)
        for idx2 in xrange(idx1, max_dim):
            z_probs[idx2] += val
        z_probs[idx1] = exp(-z_probs[idx1])
    return z_probs

def compute_p_z_given_w(input_embedding, context_embeddings):
    n = len(context_embeddings)
    d = len(context_embeddings[0])
    p_z_given_w = np.zeros(d)
    for context_vec in context_embeddings:
        p_z_given_w += compute_unnorm_z_probs_recursively(input_embedding, context_vec, d)
    return p_z_given_w / p_z_given_w.sum()

def get_nearest_neighbors(input_embedding, in_word_idx, context_embeddings, z, k):
    sparsity_weight = 0.001
    dim_penalty = 1.1
    input_embedding = np.array(input_embedding[:z])
    scores = np.zeros(len(context_embeddings))
    for idx, context_embed in enumerate(context_embeddings):
        context_embed = np.array(context_embed[:z])
        scores[idx] = np.dot(input_embedding, np.array(context_embed[:z]))
    scores[in_word_idx] = -100000
    return np.argsort(-scores)[:k]

if __name__ == '__main__':
    # some hardcoded var, make as input args later
    input_embedding_file = "/Users/enalisnick/Dropbox/iW2V-models/text8_output_single_thread_grad_fix.txt"
    context_embedding_file = "/Users/enalisnick/Dropbox/iW2V-models/text8_context_output_single_thread_grad_fix.txt"
    k = 10000
    word_to_plot = "apple"
    num_of_modes_to_plot = 5
    num_of_nns_to_get = 5

    print "loading embeddings and vocabulary..."
    in_vocab, in_embeddings = read_embedding_file(input_embedding_file)
    in_vocab = in_vocab[:k]
    in_embeddings = in_embeddings[:k]
    out_vocab, out_embeddings = read_embedding_file(context_embedding_file)
    out_vocab = out_vocab[:k]
    out_embeddings = out_embeddings[:k]
    d = len(in_embeddings[0])
    in_word_idx = in_vocab.index(word_to_plot)

    # compute p(z | w)
    print "computing p(z | w)..."
    word_in_embedding = in_embeddings[in_word_idx]
    p_z_w = compute_p_z_given_w(word_in_embedding, out_embeddings)

    # NNs look best with dot on input-to-input embeddings
    out_embeddings = in_embeddings

    # find nearest neighbors at the modes
    sorted_prob_idx = np.argsort(-1*p_z_w) # negative one so the sort is descending
    nns_at_modes = []
    modes_used = []
    idx = 0
    while num_of_modes_to_plot > 0:
        current_idx = sorted_prob_idx[idx]
        # check if this idx is too close to previous ones
        mode_flag = False
        if (current_idx==0 and p_z_w[current_idx]>p_z_w[current_idx+1]) or (current_idx==d-1 and p_z_w[current_idx]>p_z_w[current_idx-1]) or (p_z_w[current_idx]>p_z_w[current_idx-1] and p_z_w[current_idx]>p_z_w[current_idx+1]): 
            mode_flag = True
            for mode in modes_used:
                if abs(mode[0]-current_idx) <= 5:
                    mode_flag = False
        if mode_flag:
            # get nearest neighbors at current idx
            modes_used.append((current_idx, p_z_w[current_idx]))
            nns_at_modes.append([out_vocab[j] for j in get_nearest_neighbors(word_in_embedding, in_word_idx, out_embeddings, current_idx+1, num_of_nns_to_get).tolist()])
            num_of_modes_to_plot -= 1
        idx += 1
        if idx >= d:
            break
        
    # plotting the distribution
    plt.plot()
    plt.bar([x+1 for x in range(d)], p_z_w, width=1.0, facecolor='blue', edgecolor="blue")
    
    # plot the nearest neighbors at the modes
    for mode_loc, mode_nns in zip(modes_used, nns_at_modes):
        plt.annotate(', '.join(mode_nns), xy=(mode_loc[0]+1, mode_loc[1]+0.001),  xycoords='data',
                xytext=(mode_loc[0]+5, mode_loc[1]+0.005), arrowprops=dict(facecolor='black', shrink=0.05, frac=0.1, headwidth=2, width=1))
    plt.title("p(z|w="+word_to_plot+")")
    plt.xlim(1,d)
    plt.ylim(0,modes_used[0][1]+0.007)
    plt.savefig("p_z_w_"+word_to_plot+".png")
