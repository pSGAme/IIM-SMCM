#!/bin/bash


# ImageNet
topk=$1 # 10

dataset="imagenet100"
trainer="locoop"
part="all"



#CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh imagenet vit_b16_ep50 ${topk} new output/imagenet/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk60/seed1_new_lr2.5e-4_ep30
#CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh imagenet vit_b16_ep50 ${topk} new output/imagenet/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk60/seed2_new_lr2.5e-4_ep30
#CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh imagenet vit_b16_ep50 ${topk} new output/imagenet/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk60/seed3_new_lr2.5e-4_ep30
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh imagenet vit_b16_ep50 ${topk} base output/imagenet/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk60/seed1_base_lr2.5e-4_ep30
#CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh imagenet vit_b16_ep50 ${topk} base output/imagenet/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk60/seed2_base_lr2.5e-4_ep30
#CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh imagenet vit_b16_ep50 ${topk} base output/imagenet/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk60/seed3_base_lr2.5e-4_ep30
#


#
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed1_all
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed2_all
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_4shots/topk50/seed3_all

CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed1_all
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed2_all
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_8shots/topk50/seed3_all

CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed1_all
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed2_all
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/eval.sh imagenet100 vit_b16_ep30 ${topk} all output/imagenet100/LOCALPROMPT/vit_b16_ep30_16shots/topk50/seed3_all

CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_4shots/lambda0.25_topk40/seed1_${part}
CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_4shots/lambda0.25_topk40/seed2_${part}
CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_4shots/lambda0.25_topk40/seed3_${part}

CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_8shots/lambda0.25_topk40/seed1_${part}
CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_8shots/lambda0.25_topk40/seed2_${part}
CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_8shots/lambda0.25_topk40/seed3_${part}

CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk40/seed1_${part}
CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk40/seed2_${part}
CUDA_VISIBLE_DEVICES=0 sh scripts/${trainer}/eval.sh ${dataset} vit_b16_ep50 ${topk} ${part} output/${dataset}/LoCoOp/vit_b16_ep50_16shots/lambda0.25_topk40/seed3_${part}