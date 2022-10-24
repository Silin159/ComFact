#!/bin/bash

# set path to dataset here
lm="deberta-large"
portion="all"
window="nlg"
task="rel_tail"
eval_set="test"
num_gpus=4
visible=0,1,2,3

CUDA_VISIBLE_DEVICES=${visible} python -m torch.distributed.launch \
       --nproc_per_node ${num_gpus} baseline.py \
       --params_file baseline/configs/params-${lm}.json \
       --dataroot data/${portion}/${task}/${window} \
       --exp_name ${portion}-${lm}-${window}-${task}-${eval_set} \
       --eval_dataset ${eval_set}
