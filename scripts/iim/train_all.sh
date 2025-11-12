#!/bin/bash

DATASET=imagenet100
CFG=vit_b16_sgd_bs32_rr0.65_ep50_lr_2.5e-4   # config file
SHOTS=4   # number of shots (1, 2, 4, 8, 16) 4
lambda=5
div_value=0.5
topk=50
part=all
num_neg=300
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 4 $lambda $div_value $topk $part $num_neg
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 16 $lambda $div_value $topk $part $num_neg

#DATASET=imagenet100
#CFG=vit_b16_sgd_bs32_rr0.5_ep50_lr_2.5e-4   # config file
#SHOTS=4   # number of shots (1, 2, 4, 8, 16) 4
#lambda=5 # 5
#div_value=0.5 # 0.5
#topk=50 # 50
#part=all
#num_neg=60
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 4 $lambda $div_value $topk $part $num_neg
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 16 $lambda $div_value $topk $part $num_neg





