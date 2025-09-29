#!/bin/bash

topk=$1

for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 2.0
do
  CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/alpha/seed1_new_bs32 300 ${alpha}
  CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/alpha/seed2_new_bs32 300 ${alpha}
  CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/eval.sh imagenet vit_b16_ep10_lr_2.5e-4 ${topk} new output/imagenet/ProSimO/vit_b16_ep10_lr_2.5e-4_4shots/alpha/seed3_new_bs32 300 ${alpha}
done