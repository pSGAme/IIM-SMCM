#!/bin/bash

# custom config
TRAINER=LoCoOp
DATA="/data/ood/"

DATASET=$1
CFG=$2
topk=$3
part=$4 # base, new, all
MODEL_dir=$5
Output_dir="${MODEL_dir}/eval"

#python eval_ood_detection.py \
#--root ${DATA} \
#--trainer ${TRAINER} \
#--dataset-config-file configs/datasets/${DATASET}.yaml \
#--config-file configs/trainers/LoCoOp/${CFG}.yaml \
#--output-dir ${Output_dir} \
#--model-dir ${MODEL_dir} \
#--load-epoch 50 \
#--T ${T} \

python eval_ood_detection.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--in_dataset ${DATASET} \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--load-epoch 50 \
--output-dir ${Output_dir} \
--model-dir ${MODEL_dir} \
--top_k ${topk} \
DATASET.SUBSAMPLE_CLASSES ${part}