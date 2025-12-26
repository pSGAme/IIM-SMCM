#!/bin/bash

topk=50
dataset="imagenet"
part=new
num_neg=300
shot=4
CFG=common
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg300_topk50_lamda5_div0.5"
alpha=1.0 # the weight of S-MCM

for seed in 1 2 3
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done

