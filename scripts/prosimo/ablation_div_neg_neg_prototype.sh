#!/bin/bash
TRAINER=ProSimO



# div_value
CUDA_VISIBLE_DEVICES=1 sh scripts/prosimo/train.sh imagenet vit_b16_ep10_lr_2.5e-4 4 0.0 0.0 50 new 300


