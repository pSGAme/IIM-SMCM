##!/bin/bash
#
topk=$1
#
#
#num_neg=10
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed1_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed2_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed3_new_bs32 ${num_neg}
#
#num_neg=50
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed1_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed2_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed3_new_bs32 ${num_neg}
#
#num_neg=100
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed1_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed2_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed3_new_bs32 ${num_neg}
#
#num_neg=200
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed1_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed2_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed3_new_bs32 ${num_neg}
#
#num_neg=300
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed1_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed2_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed3_new_bs32 ${num_neg}
#
#num_neg=400
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed1_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed2_new_bs32 ${num_neg}
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda5_div0.5/seed3_new_bs32 ${num_neg}
#
num_neg=300

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda0.0_div0.0/seed1_new_bs32 ${num_neg}

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda0.0_div0.0/seed2_new_bs32 ${num_neg}

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/numneg${num_neg}_topk50_lamda0.0_div0.0/seed3_new_bs32 ${num_neg}

