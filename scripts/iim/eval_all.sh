#!/bin/bash

topk=50
dataset="imagenet"
part="new"
num_neg=300
shot=4
CFG=vit_b16_sgd_bs32_rr0.08_ep30_lr_3.5e-4
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.1/"
alpha=0.9

for seed in 1 2 3
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done