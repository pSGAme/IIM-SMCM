#!/bin/bash
# custom config
TRAINER=LOCALPROMPT

DATA="/data/ood/"
DATASET=$1
CFG=$2  # config file
SHOTS=$3   # number of shots (1, 2, 4, 8, 16) 4
lambda=$4 # 5
div_value=$5 # 0.5
topk=$6 # 50
part=$7

for SEED in  1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/topk${topk}/seed${SEED}_${part}_bs32_epoch50
#    if [ -d "$DIR" ]; then
#        echo "Oops! The results exist at ${DIR} (so skip this job)"
#    else
      echo $PWD
      python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      --lambda_value ${lambda} \
      --div_value ${div_value} \
      --topk ${topk} \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES ${part}
    #fi
done

#
#CUDA_VISIBLE_DEVICES=1 sh scripts/train.sh imagenet vit_b16_ep30 4 5 0.5 50 base