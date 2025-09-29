#!/bin/bash

TRAINER=LOCALPROMPT
DATA="/data/ood/"

DATASET=$1
CFG=$2
topk=$3 # 10
part=$4 # base, new, all
MODEL_dir=$5

Output_dir="${MODEL_dir}/eval2"

python eval_ood_detection.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--in_dataset ${DATASET} \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--load-epoch 30 \
--output-dir ${Output_dir} \
--model-dir ${MODEL_dir} \
--top_k ${topk} \
DATASET.SUBSAMPLE_CLASSES ${part}

# For local-prompt
#CUDA_VISIBLE_DEVICES=1 sh scripts/eval.sh data imagenet vit_b16_ep30 16 10 base output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/nctx16_cscTrue_ctpend_topk50/seed1_base


# For ProSimO
#CUDA_VISIBLE_DEVICES=1 sh scripts/eval.sh data imagenet vit_b16_ep30 16 50 base output/imagenet/ProSimO/vit_b16_ep30_4shots/nctx16_cscTrue_ctpend_topk50x2_nocrop_lr0.00025_ep10_text_proj_true/seed2_base
