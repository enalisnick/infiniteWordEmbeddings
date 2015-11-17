

WWSI_FILE = 'Evaluation/wwsi/wwsi.txt' 

# Return ground truth and
# examples in form of list of (word_idx, context_idxs) 
def read_wwsi_file(wwsi_file=WWSI_FILE):
  with open(wwsi_file) as f:
    # Read word and number of meanings 
    _,word,_,num_meanings = f.readline().split()
   
    # Read all sentences for each meaning 
    for i in range(0,num_meanings):
      _,_,num_sentences = f.readline().split()
      for j in range(0,num_sentences):
        
       
if __name__ == '__main__':
  
