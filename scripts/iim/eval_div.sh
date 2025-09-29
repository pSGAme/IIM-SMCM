#!/bin/bash

topk=$1

## div = 0
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.0/seed1_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.0/seed2_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.0/seed3_new_bs32


## div = 0.1
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.1/seed1_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.1/seed2_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.1/seed3_new_bs32

##div = 0.5
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.5/seed1_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.5/seed2_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.5/seed3_new_bs32

## div = 1.0
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div1.0/seed1_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div1.0/seed2_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div1.0/seed3_new_bs32

## div = 2.0
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div2.0/seed1_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div2.0/seed2_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div2.0/seed3_new_bs32

## div = 5.0
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div5.0/seed1_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div5.0/seed2_new_bs32

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div5.0/seed3_new_bs32
