#!/bin/bash

topk=50
dataset="imagenet"
part=new
num_neg=300
shot=4
CFG=test
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
alpha=1.0

for seed in 1 2 3
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done
#
#shot=16
#modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
#
#for seed in 1 2 3
#do
#  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
#done
#

