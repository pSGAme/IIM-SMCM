#!/bin/bash

topk=50
dataset="imagenet"
part="new"
config="TEST"
shot=4
num_neg=300

model_dir="output/${dataset}/ProSimO/${config}_${shot}shots/numneg${num_neg}_topk${topk}_lamda5_div0.5"
alpha=0.9

for seed in  2 3
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh ${dataset}  ${config} ${topk} ${part} ${model_dir}/seed${seed}_${part}_bs32 ${num_neg} ${alpha}
done
