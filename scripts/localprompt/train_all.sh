#!/bin/bash
# custom config
TRAINER=LOCALPROMPT


config_file="rebuttal"
CUDA_VISIBLE_DEVICES=1 sh scripts/localprompt/train.sh imagenet ${config_file} 4 5 0.5 50 new
#CUDA_VISIBLE_DEVICES=1 sh scripts/localprompt/train.sh imagenet ${config_file} 16 5 0.5 50 new