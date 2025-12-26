#!/bin/bash


## common set
topk=50
lambda=5
num_neg=300
div_value=0.5

## For ImageNet-1k
DATASET=imagenet
part=all
CFG=common  # config file

CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 4 $lambda $div_value $topk $part $num_neg  # 4 shot
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 16 $lambda $div_value $topk $part $num_neg  # 16 shot

## For ImageNet-500-Easy
part=base
CFG=common  # config file
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 4 $lambda $div_value $topk $part $num_neg  # 4 shot
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 16 $lambda $div_value $topk $part $num_neg  # 16 shot

## For ImageNet-500-Hard
part=new
CFG=ImageNet-500-Hard  # config file
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 4 $lambda $div_value $topk $part $num_neg  # 4 shot
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 16 $lambda $div_value $topk $part $num_neg  # 16 shot


## For ImageNet-100
DATASET=imagenet100
part=all
CFG=common  # config file
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 4 $lambda $div_value $topk $part $num_neg  # 4 shot
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh $DATASET $CFG 16 $lambda $div_value $topk $part $num_neg  # 16 shot






