#!/bin/bash

DATASET=imagenet
CFG=vit_b16_sgd_bs32_rr0.08_ep30_lr_3.5e-4   # config file
SHOTS=4   # number of shots (1, 2, 4, 8, 16) 4
lambda=5 # 5
div_value=0.1 # 0.5
topk=50 # 50
part=new
num_neg=300
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG $SHOTS $lambda $div_value $topk $part $num_neg


#CFG=vit_b16_adamw_bs32_rr0.08_ep50_lr_2.5e-4_biproj  # config file
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG $SHOTS $lambda $div_value $topk $part $num_neg