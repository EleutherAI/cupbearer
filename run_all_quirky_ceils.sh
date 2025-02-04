#!/bin/zsh

# Define the original and temporary script paths
original_script="src/cupbearer/scripts/quirky_ceilings.py"
temp_script="/tmp/quirky_ceilings.py"

datasets=("hemisphere" "capitals"  "population" "sciq" "sentiment" "nli" "authors" "addition" "subtraction" "multiplication" "modularaddition" "squaring")

# Copy the original script to a temporary location
cp $original_script $temp_script

for DSET in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python $temp_script --dataset $DSET #--attribution --ablation pcs --mlp
done

# Optionally, remove the temporary script after execution
rm $temp_script