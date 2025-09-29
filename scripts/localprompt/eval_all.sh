#!/bin/bash


# ImageNet
topk=10 # 10
dataset="imagenet"
config="vit_b16_ep30" # specify the backbone
part="new"
modeldir="output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/topk50"

for seed in 1
do
  CUDA_VISIBLE_DEVICES=1 sh scripts/localprompt/eval.sh ${dataset} ${config} ${topk} ${part} ${modeldir}/seed${seed}_${part}
done
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed1_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed2_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed3_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed1_new
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed2_new
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed3_new
#
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed1_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed2_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed3_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed1_new
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed2_new
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed3_new
#
#
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed1_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed2_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} base output/imagenet/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed3_base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed1_new
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed2_new
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet vit_b16_ep30 ${topk} new output/imagenet/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed3_new
#
#
