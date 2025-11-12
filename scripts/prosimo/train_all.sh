#!/bin/bash
TRAINER=ProSimO

config_file="TEST"
top_k=50

CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet ${config_file} 4 5 0.5 ${top_k} base 300
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet ${config_file} 16 5 0.5 ${top_k} base 300




# CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet ${config_file} 16 5 0.5 12 new 300


#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 base

#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 8 5 0.5 50 base
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 8 5 0.5 50 new
#
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 16 5 0.5 50 base
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 16 5 0.5 50 new


#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet100 vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 all
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet100 vit_b16_ep10_lr_2.5e-4 8 5 0.5 50 all
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet100 vit_b16_ep10_lr_2.5e-4 16 5 0.5 50 all



#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet100 vit_b16_ep10_lr_2.5e-4 16 5 0.5 50 all 60
