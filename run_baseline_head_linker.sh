#!/bin/bash

lm="roberta-large"
portion="persona"
window="nlg"

# Prepare directories for intermediate results of each subtask
mkdir -p pred/${portion}-${lm}-${window}-pipeline-test
visible=0

CUDA_VISIBLE_DEVICES=${visible} python baseline.py \
   --eval_only \
   --checkpoint runs/${portion}-${lm}-${window}-head-test/ \
   --params_file runs/${portion}-${lm}-${window}-head-test/params-${lm}.json \
   --eval_dataset test \
   --dataroot data/${portion}/head/${window} \
   --output_file pred/${portion}-${lm}-${window}-head-test/predictions.json

CUDA_VISIBLE_DEVICES=${visible} python data_preprocessing_pipeline_test.py \
   --model ${lm} \
   --window ${window} \
   --portion ${portion}

CUDA_VISIBLE_DEVICES=${visible} python evaluate_pipeline.py \
   --model ${lm} \
   --window ${window} \
   --portion ${portion} \
   --linking head
