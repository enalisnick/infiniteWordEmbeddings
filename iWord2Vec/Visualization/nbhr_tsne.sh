#!/bin/bash

# ARGUMENTS 
# 1. Model name
# 2. Model location
# 3. Number of neighbors

OUTPUT_PATH="Results/Visualization/$1"
$(mkdir -p "$OUTPUT_PATH")

words=( net apple france paris plant )
model=$2
K=$3
num_dims=( 20 40 60 80 100 140 180 200 )

for w in "${words[@]}" 
do
  output_path="$OUTPUT_PATH/$w/"
  echo "$output_path"
  mkdir -p $output_path
  
  for num_dim in  "${num_dims[@]}" 
  do
    echo "running with params $model $w $K $num_dim $output_path"
    python Visualization/nbhr_tsne.py $model $w $K $num_dim $output_path 
  done
done
:
