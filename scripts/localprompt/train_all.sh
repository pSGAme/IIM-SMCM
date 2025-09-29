#!/bin/bash
# custom config
TRAINER=LOCALPROMPT


config_file="rn50_ep30"
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/train.sh imagenet ${config_file} 4 5 0.5 12 new
CUDA_VISIBLE_DEVICES=0 sh scripts/localprompt/train.sh imagenet ${config_file} 16 5 0.5 12 new