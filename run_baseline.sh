#!/bin/bash

lm="deberta-large"
portion="persona"
window="nlg"
task="fact_full"
eval_set="test"

# Prepare directories for intermediate results of each subtask
mkdir -p pred/${portion}-${lm}-${window}-${task}-${eval_set}
visible=0

CUDA_VISIBLE_DEVICES=${visible} python baseline.py \
   --eval_only \
   --checkpoint runs/${portion}-${lm}-${window}-${task}-${eval_set}/ \
   --params_file runs/${portion}-${lm}-${window}-${task}-${eval_set}/params-${lm}.json \
   --eval_dataset ${eval_set} \
   --dataroot data/${portion}/${task}/${window} \
   --output_file pred/${portion}-${lm}-${window}-${task}-${eval_set}/predictions.json
