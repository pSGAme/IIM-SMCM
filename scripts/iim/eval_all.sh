#!/bin/bash

topk=50
dataset="imagenet"
part="new"
num_neg=300
shot=4
modeldir="output/${dataset}/IIM/vit_b16_ep20_lr_2.5e-4_${shot}shots/numneg${num_neg}_topk50_lamda5_div0.1/"
alpha=0.9

for seed in 2
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/iim/eval.sh ${dataset} vit_b16_ep10_lr_2.5e-4 ${topk} ${part} ${modeldir}/seed${seed}_${part}_bs32 ${num_neg} ${alpha}
done


#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 50 all output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed1_all_bs32

#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed1_base_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed2_base_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed3_base_bs32
#
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed1_new_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed2_new_bs32
##
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed3_new_bs32
##
##
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed1_base_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed2_base_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed3_base_bs32
#
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed1_new_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed2_new_bs32
##
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed3_new_bs32
#
##
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed1_base_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed2_base_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed3_base_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed1_new_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed2_new_bs32
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed3_new_bs32

#
# ImageNet-100
#topk=$1
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed1_all_bs32
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed2_all_bs32
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed3_all_bs32


#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed1_all_bs32
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed2_all_bs32
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_16shots/topk50_text_proj/seed3_all_bs32
#
#
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed1_all_bs32
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed2_all_bs32
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimO/vit_b16_ep10_lr_2.5e-4_8shots/topk50_text_proj/seed3_all_bs32