import numpy as np
import matplotlib                                                               
matplotlib.use('Agg')                                                           
import matplotlib.pyplot as plt 
import sys, getopt

from Auto_Eval.auto_eval_iSG import process_embeddings_dir
from Evaluation.eval_lib import read_embedding_file 

def heatmap(data, labels):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.jet, alpha=0.8)
    ax.set_yticklabels(labels, minor=False)

    fig.colorbar(heatmap, ax=ax)
    return fig, ax

def plot(vocab, embeddings, words_to_plot, filename=""):
 
    data = np.zeros((len(words_to_plot), len(embeddings[0]))) 
    labels = []
    for idx, word_to_plot in enumerate(words_to_plot):
      in_word_idx = vocab.index(word_to_plot)
      data[idx] = in_embeddings_all[in_word_idx]
      labels.append(word_to_plot)

    f, ax = heatmap(data, labels)
    # save figure                                                               
    f.savefig(filename)            

if __name__ == '__main__':
    k = 15000 # truncate the vocabulary to the top k most frequent words
    sparsity = 0.0
    dim_penalty = 0.0
    help_message = 'graph_p_z.py -r <path to directory containing embedding files> -w <comma separated list of words to plot> -s <optional comma separated list of words to marginalize over>'

    # read in command line arguments
    input_embedding_file = ""
    context_embedding_file = ""
    words_to_plot = ['photon', 'kwanzaa', 'william', 'bank', 'race', 'play']
   
    # filename
    filename = "" 
   
    # get args
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hr:c:w:s:n:",["rootDir=","words=","filename="])
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
        elif opt in ("-n", "--filename"):
            filename = arg
 
    # extract dim and sparsity penalties from file names
    arr = process_embeddings_dir(rootDir) 
    for input_embedding_file,context_embedding_file,sparsity,dim_penalty in arr:
	print 'Input embeddings file: ', input_embedding_file
	print 'Context embeddings file: ', context_embedding_file
	print 'words_to_plot: ', ", ".join(words_to_plot)
	print 'sparsity penalty: ', str(sparsity)
	print 'dimension penalty: ', str(dim_penalty)


	print "loading embeddings and vocabulary..."
	in_vocab_all, in_embeddings_all = read_embedding_file(input_embedding_file)
	in_vocab = in_vocab_all[:k]
	in_embeddings = in_embeddings_all[:k]
	out_vocab_all, out_embeddings_all = read_embedding_file(context_embedding_file)
	out_vocab = out_vocab_all[:k]
	out_embeddings = out_embeddings_all[:k]

        plot(in_vocab_all, in_embeddings_all, words_to_plot, "heatmap_input.png")
        plot(in_vocab_all, out_embeddings_all, words_to_plot, "heatmap_context.png")
