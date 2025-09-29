#!/bin/bash

topk=$1


# neg = 0
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda0.0_div0.5/seed1_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda0.0_div0.5/seed2_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda0.0_div0.5/seed3_new_bs32

# neg = 0.1
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda0.1_div0.5/seed1_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda0.1_div0.5/seed2_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda0.1_div0.5/seed3_new_bs32

# neg = 1.0
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda1.0_div0.5/seed1_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda1.0_div0.5/seed2_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda1.0_div0.5/seed3_new_bs32

# neg = 2.0
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda2.0_div0.5/seed1_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda2.0_div0.5/seed2_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda2.0_div0.5/seed3_new_bs32


# neg = 5
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5.0_div0.5/seed1_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5.0_div0.5/seed2_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5.0_div0.5/seed3_new_bs32


# neg = 10.0
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda10.0_div0.5/seed1_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda10.0_div0.5/seed2_new_bs32

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda10.0_div0.5/seed3_new_bs32