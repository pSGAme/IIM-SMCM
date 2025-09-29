#!/bin/bash

DEVICE=1

for topk in 1 5 10 20 50 80
do
    # k = 1
    # CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk1_lamda5_div0.5/seed1_new_bs32

    # CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk1_lamda5_div0.5/seed2_new_bs32

#    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk1_lamda5_div0.5/seed3_new_bs32
#
#    # k = 5
#    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk5_lamda5_div0.5/seed1_new_bs32
#
#    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk5_lamda5_div0.5/seed2_new_bs32
#
#    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk5_lamda5_div0.5/seed3_new_bs32
#
#
#    # k = 10
#    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk10_lamda5_div0.5/seed1_new_bs32
#
#    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk10_lamda5_div0.5/seed2_new_bs32
#
#    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk10_lamda5_div0.5/seed3_new_bs32

    # k = 20
    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk20_lamda5_div0.5/seed1_new_bs32

    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk20_lamda5_div0.5/seed2_new_bs32

    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk20_lamda5_div0.5/seed3_new_bs32

    # k = 50
    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.5/seed1_new_bs32

    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.5/seed2_new_bs32

    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk50_lamda5_div0.5/seed3_new_bs32

    # k = 80
    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk80_lamda5_div0.5/seed1_new_bs32

    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk80_lamda5_div0.5/seed2_new_bs32

    CUDA_VISIBLE_DEVICES=${DEVICE} sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/topk80_lamda5_div0.5/seed3_new_bs32
done
