#!/bin/bash

topk=50
dataset="imagenet"
part="new"
config="vit_b16_ep10_lr_2.5e-4"
shot=4
modeldir="output/${dataset}/ProSimO/${config}_${shot}shots/numneg300_topk50_lamda5_div0.5/"
num_neg=300
alpha=0.9

for seed in 1
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh ${dataset}  ${config} ${topk} ${part} ${modeldir}/seed${seed}_${part}_bs32 ${num_neg} ${alpha}
done
