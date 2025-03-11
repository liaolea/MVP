#!/bin/bash

DATA=/your/path
TRAINER=PromptCLIP
SHOTS=16
export CUDA_VISIBLE_DEVICES="0"

DATASET=$1
CFG=$2

declare -A type3
type3['oxford_pets']="a photo of a {}, a type of pet."
type3['oxford_flowers']="a photo of a {}, a type of flower."
type3['fgvc_aircraft']="a photo of a {}, a type of aircraft."
type3['dtd']="a photo of {}, a type of texture."
type3['eurosat']="a photo of {}, a type of satellite imagery."
type3['stanford_cars']="a photo of a {}, a type of car."
type3['food101']="a photo of {}, a type of food."
type3['sun397']="a photo of a {}, a type of scene."
type3['caltech101']="a photo of a {}, a type of entity."
type3['ucf101']="a photo of a person doing {}, a type of human activity."
type3['imagenet']="a photo of a {}, a type of entity."

original_prompts=("a photo of a {}." "a photo of {}." "a clip of a {}." "a frame of a {}." "I take a photo of a {}." "We take a photo of a {}." "She takes a photo of a {}." "He takes a photo of a {}." "They take a photo of a {}." "You take a photo of a {}." "I will take a photo of a {}." "I took a photo of a {}." "a serene photo of a {}." "a radiant photo of a {}." "a joyful photo of a {}." "a somber photo of a {}." "a ominous photo of a {}." "a melancholic photo of a {}.")

if [[ -z "${type3[$DATASET]}" ]]; then
    echo "No Dataset. "
    PROMPTS=("${original_prompts[@]}")
else
    PROMPTS=("${original_prompts[@]}" "${type3[$DATASET]}")
fi

for SEED in 1 2 3
do
    for PROMPT in "${PROMPTS[@]}"
    do
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset_name ${DATASET} \
        --prompt "${PROMPT}" \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ablation/output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/prompt_${PROMPT// /_} \
        --model-dir ablation/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots \
        --load-epoch 100 \
        --eval-only
    done
done
