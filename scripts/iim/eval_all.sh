#!/bin/bash
#
#topk=50
#dataset="imagenet100"
#part=all
#num_neg=300
#shot=16
#CFG=vit_b16_sgd_bs32_rr0.4_ep50_lr_2.5e-4
#modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
#alpha=0.9
#
#for seed in 1 2 3
#do
#  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
#done
#

topk=50
dataset="imagenet"
part=all
num_neg=300
shot=4
CFG=vit_b16_sgd_bs32_rr0.85_ep50_lr_2.5e-4
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
alpha=1.0

for seed in 1
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done

topk=50
dataset="imagenet"
part=all
num_neg=300
shot=4
CFG=vit_b16_sgd_bs32_rr0.85_ep50_lr_2.5e-4
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
alpha=1.1

for seed in 1
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done


topk=50
dataset="imagenet"
part=all
num_neg=300
shot=4
CFG=vit_b16_sgd_bs32_rr0.85_ep50_lr_2.5e-4
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
alpha=1.2

for seed in 1
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done


topk=50
dataset="imagenet"
part=all
num_neg=300
shot=4
CFG=vit_b16_sgd_bs32_rr0.85_ep50_lr_2.5e-4
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
alpha=1.3

for seed in 1
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done



topk=50
dataset="imagenet"
part=all
num_neg=300
shot=4
CFG=vit_b16_sgd_bs32_rr0.85_ep50_lr_2.5e-4
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
alpha=1.4

for seed in 1
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done


topk=50
dataset="imagenet"
part=all
num_neg=300
shot=4
CFG=vit_b16_sgd_bs32_rr0.85_ep50_lr_2.5e-4
modeldir="output/${dataset}/IIM/${CFG}_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.5"
alpha=1.5

for seed in 1
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} ${CFG} ${topk} ${part} ${modeldir}/seed${seed}_${part} ${num_neg} ${alpha}
done



