#!/bin/bash
TRAINER=ProSimO


# neg_value
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 0.0 0.5 50 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 0.1 0.5 50 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 1.0 0.5 50 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 2.0 0.5 50 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5.0 0.5 50 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 10.0 0.5 50 new


## div_value
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.0 50 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.1 50 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 1.0 50 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 2.0 50 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 5.0 50 new
#
## train_k
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 10 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 20 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 80 new

