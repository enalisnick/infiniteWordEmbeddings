#!/bin/bash
### USER DEFINED VARIABLES ###
trainFile="train.txt"
initSize=5
windowSize=10
negativeSamps=5
iters=5
learningRate=0.5
dimPenalty=10
sparsityWeight=0.001
### LOG THE PARAMS ###
baseFileName="$trainFile-$initSize-$windowSize-$negativeSamps-$iters-$learningRate-$dimPenalty-$sparsityWeight"
logFileName="$baseFileName.log"
vecsFileName="$baseFileName.vecs"
datetime=$(date +"%m-%d-%Y, %H:%M:%S")
echo "Ran on: $datetime" >> "Results/$logFileName"
echo "# TRAINING SETTINGS #" >> "Results/$logFileName"
echo "Train Corpus: $trainFile" >> "Results/$logFileName"
echo "Initial dimensionality: $initSize" >> "Results/$logFileName"
echo "Context window size: $windowSize" >> "Results/$logFileName"
echo "Num. of negative samples: $negativeSamps" >> "Results/$logFileName"
echo "Training iterations (epochs): $iters" >> "Results/$logFileName"
echo "Learning rate: $learningRate" >> "Results/$logFileName"
echo "Per dimension penalty: $dimPenalty" >> "Results/$logFileName"
echo "Weight on sparsity penalty: $sparsityWeight" >> "Results/$logFileName"
### TRAIN iW2V ###
./iW2V -train $train_file -output $vecsFileName -initSize $initSize -maxSize 2000 -window $windowSize -sample 1e-4 -negative $negativeSamps -iter $iters -alpha $learningRate -dimPenalty $dimPenalty -sparsityWeight $sparsityWeight -threads 10
### EVAL ON WORD_SIM353  ###
spearmanCor=$(python Evaluation/wordSim353_eval.py $vecsFileName)
echo "# EVALUATION: WORD SEMANTIC SIMILARITY #" >> "Results/$logFileName"
echo "Spearman's Rank Correlation Coefficient on WordSim353: $spearmanCor" >> "Results/$logFileName"
### PRINT SOME NEAREST NEIGHBORS ###
#echo "# EVALUATION: NEAREST NEIGHBORS #" >> "Results/$logFileName"
echo "" >> "Results/$logFileName" 

