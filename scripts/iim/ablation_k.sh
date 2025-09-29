#!/bin/bash
TRAINER=ProSimO


# neg_value

## train_k
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 1 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 5 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 10 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 20 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 80 new

