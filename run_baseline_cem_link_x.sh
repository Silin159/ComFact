#!/bin/bash

lm="deberta-large"
portion="all"
window="nlg"
task="rel_tail"
eval_set="test"
root="runs"

# Prepare directories for intermediate results of each subtask
mkdir -p pred/${portion}-${lm}-${window}-${task}-cem
visible=0

CUDA_VISIBLE_DEVICES=${visible} python baseline.py \
   --eval_only \
   --checkpoint ${root}/${portion}-${lm}-${window}-${task}-${eval_set}/ \
   --params_file ${root}/${portion}-${lm}-${window}-${task}-${eval_set}/params-${lm}.json \
   --eval_dataset ${eval_set} \
   --dataroot data/cem/${task}/${window} \
   --output_file pred/${portion}-${lm}-${window}-${task}-cem/predictions.json
