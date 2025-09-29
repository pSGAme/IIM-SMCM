#!/bin/bash
TRAINER=ProSimO



# div_value

CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new 400 # 23782
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new 10
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new 50
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new 100
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new 200
CUDA_VISIBLE_DEVICES=0 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 5 0.5 50 new 300






