#!/bin/bash


#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet vit_b16_ep50_lr_2.5e-4 4 5 0.1 50 base 300
CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet vit_b16_ep20_lr_2.5e-4 4 5 0.1 50 new 300
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet vit_b16_ep50_lr_2.5e-4 8 5 0.1 50 base 300
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet vit_b16_ep50_lr_2.5e-4 8 5 0.1 50 new 300
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet vit_b16_ep50_lr_2.5e-4 16 5 0.1 50 base 300
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet vit_b16_ep50_lr_2.5e-4 16 5 0.1 50 new 300
#
#
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet100 vit_b16_ep50_lr_2.5e-4 4 5 0.1 50 all 60
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet100 vit_b16_ep50_lr_2.5e-4 8 5 0.1 50 all 60
#CUDA_VISIBLE_DEVICES=1 sh scripts/iim/train.sh imagenet100 vit_b16_ep50_lr_2.5e-4 16 5 0.1 50 all 60
