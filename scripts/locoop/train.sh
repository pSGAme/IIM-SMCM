#!/bin/bash
# custom config

TRAINER=LoCoOp

DATA="/data/ood/"
DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
lambda=$4
topk=$5
part=$6

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/lambda${lambda}_topk${topk}/seed${SEED}_${part}
#    if [ -d "$DIR" ]; then
#        echo "Oops! The results exist at ${DIR} (so skip this job)"
#    else
        echo $PWD
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/LoCoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        --lambda_value ${lambda} \
        --topk ${topk} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${part}
#    fi
done

# imagenet-500-hard

#CUDA_VISIBLE_DEVICES=0 sh scripts/locoop/train.sh imagenet vit_b16_ep50 16  0.25 200 new
