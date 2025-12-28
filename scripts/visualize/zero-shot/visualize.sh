#!/bin/bash

# custom config

TRAINER=ZeroShot
DATA="/data/ood/"
CFG="null"

DATASET=$1
topk=$2 # 10
part=$3 # base, new, all



MODEL_dir=output/${DATASET}/${TRAINER}/${part}

Output_dir="${MODEL_dir}/eval"

python visualize_imagenet1k.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--in_dataset ${DATASET} \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${Output_dir} \
--top_k ${topk} \
DATASET.SUBSAMPLE_CLASSES ${part} # args.opts


