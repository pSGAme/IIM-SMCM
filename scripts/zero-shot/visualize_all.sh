#!/bin/bash


CUDA_VISIBLE_DEVICES=1 sh scripts/zero-shot/visualize.sh imagenet 10 all
# CUDA_VISIBLE_DEVICES=0 sh scripts/zero-shot/eval.sh imagenet ${topk} new
# CUDA_VISIBLE_DEVICES=0 sh scripts/zero-shot/eval.sh imagenet ${topk} base
# CUDA_VISIBLE_DEVICES=0 sh scripts/zero-shot/eval.sh imagenet100 ${topk} all
