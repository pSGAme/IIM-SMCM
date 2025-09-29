#!/bin/bash

topk=$1
CUDA_VISIBLE_DEVICES=0 sh scripts/zero-shot/eval.sh imagenet100 ${topk} all
# CUDA_VISIBLE_DEVICES=0 sh scripts/zero-shot/eval.sh imagenet ${topk} new
# CUDA_VISIBLE_DEVICES=0 sh scripts/zero-shot/eval.sh imagenet ${topk} base
# CUDA_VISIBLE_DEVICES=0 sh scripts/zero-shot/eval.sh imagenet100 ${topk} all
