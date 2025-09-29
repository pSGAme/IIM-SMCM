#!/bin/bash
# custom config
TRAINER=ProSimO

DATA="/data/ood/"
DATASET=$1
CFG=$2  # config file
SHOTS=$3   # number of shots (1, 2, 4, 8, 16) 4
lambda=$4 # 5
div_value=$5 # 0.5
topk=$6 # 50
part=$7
num_neg=$8

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/numneg${num_neg}_topk${topk}_lamda${lambda}_div${div_value}/seed${SEED}_${part}_bs32
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
      --num_neg_prompts ${num_neg} \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES ${part}
    #fi
done

#
#CUDA_VISIBLE_DEVICES=1 sh scripts/train.sh imagenet vit_b16_ep30 4 5 0.5 50 base