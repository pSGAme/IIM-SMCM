#!/bin/bash

# custom config
TRAINER=ProSimO
DATA="/data/ood/"

DATASET=$1
CFG=$2
topk=$3 # 10
part=$4 # base, new, all
MODEL_dir=$5
num_neg=$6
alpha=$7

Output_dir="${MODEL_dir}/eval"

python eval_ood_detection.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--in_dataset ${DATASET} \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--load-epoch 10 \
--output-dir ${Output_dir} \
--model-dir ${MODEL_dir} \
--top_k ${topk} \
--alpha ${alpha} \
--num_neg_prompts ${num_neg} \
DATASET.SUBSAMPLE_CLASSES ${part} # args.opts