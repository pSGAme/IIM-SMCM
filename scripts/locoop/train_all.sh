#!/bin/bash
# custom config


CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 4 0.25 100 new
CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 4 0.25 100 base

CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 8 0.25 100 new
CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 8 0.25 100 base

CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 16 0.25 100 new
CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 16 0.25 100 base

#
#CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 4 0.00 60 new
#CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 4 0.00 60 base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 8 0.00 60 new
#CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 8 0.00 60 base
#
#CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 16 0.00 60 new
#CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 16 0.00 60 base
