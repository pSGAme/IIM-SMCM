#!/bin/bash

topk=$1
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed1_base
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed2_base
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed3_base
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed1_new
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed2_new
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed3_new


#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj_train_vpt/seed1_base
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj_train_vpt/seed2_base
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} base output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj_train_vpt/seed3_base
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj_train_vpt/seed1_new
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj_train_vpt/seed2_new
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj_train_vpt/seed3_new

#
#topk=$1
#
# CUDA_VISIBLE_DEVICES=1 sh scripts/prosimohyper/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimOHyper/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed1_all

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimohyper/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimOHyper/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed2_all
#
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimohyper/eval.sh imagenet100 vit_b16_ep10_lr_2.5e-4 ${topk} all output/imagenet100/ProSimOHyper/vit_b16_ep10_lr_2.5e-4_4shots/topk50_text_proj/seed3_all
