#!/bin/bash

lm="roberta-large"
source_portion="persona"
target_portion="movie"
window="nlg"
task="fact_full"
eval_set="test"

# Prepare directories for intermediate results of each subtask
mkdir -p pred/cross-${lm}-${window}-${task}-${eval_set}
visible=0

CUDA_VISIBLE_DEVICES=${visible} python baseline.py \
   --eval_only \
   --checkpoint runs/${source_portion}-${lm}-${window}-${task}-${eval_set}/ \
   --params_file runs/${source_portion}-${lm}-${window}-${task}-${eval_set}/params-${lm}.json \
   --eval_dataset ${eval_set} \
   --dataroot data/${target_portion}/${task}/${window} \
   --output_file pred/cross-${lm}-${window}-${task}-${eval_set}/${source_portion}-${target_portion}-predictions.json
