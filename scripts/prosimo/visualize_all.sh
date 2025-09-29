#!/bin/bash


topk=50
dataset="imagenet"
part="all"
modeldir="output/${dataset}/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/"
num_neg=300
alpha=0.9

for seed in 1
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/visualize.sh ${dataset} vit_b16_ep10_lr_2.5e-4 ${topk} ${part} ${modeldir}seed${seed}_${part}_bs32 ${num_neg} ${alpha}
done