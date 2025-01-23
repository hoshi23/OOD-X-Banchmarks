DIR_BASE="../.."
DATA="${DIR_BASE}/data"
OUTPUT_BASE="${DIR_BASE}/output"

# config
CONFIG_BASE="LoCoOp"
CONFIG_NAME="vit_b16_ep50"
TRAINER="CoOp"
SHOTS=16


# data mode
# ImageNet-X
DATASET="imagenet_x"
SUBSAMPLE_CLASSES="custom"
CUSTOM_SPLIT_MODE="x"


# output
ADDITIONAL_SETTING="${SUBSAMPLE_CLASSES}_${CUSTOM_SPLIT_MODE}"

for SEED in 1 2 3
    do
    train_output_dir="${OUTPUT_BASE}/${TRAINER}/shots_${SHOTS}/${CONFIG_NAME}/${TRAIN_MODE}/${DATASET}/${ADDITIONAL_SETTING}/train/seed${SEED}"
    eval_output_dir="${OUTPUT_BASE}/${TRAINER}/shots_${SHOTS}/${CONFIG_NAME}/${TRAIN_MODE}/${DATASET}/${ADDITIONAL_SETTING}/eval/seed${SEED}"

    echo "Train: ImageNet-FS-X"
    python ${DIR_BASE}/src/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file ${DIR_BASE}/configs/datasets/${DATASET}.yaml \
    --config-file ${DIR_BASE}/configs/trainers/${CONFIG_BASE}/${CONFIG_NAME}.yaml \
    --ood_method mcm \
    --output-dir ${train_output_dir} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
    DATASET.ID_CLASSES_FILE "${DATA}/class_splits/imagenet/${CUSTOM_SPLIT_MODE}/id_data.txt" \
    DATASET.OOD_CLASSES_FILE "${DATA}/class_splits/imagenet/${CUSTOM_SPLIT_MODE}/ood_data.txt"


    echo "Benchmark: ImageNet-FS-X"
    python ${DIR_BASE}/src/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file ${DIR_BASE}/configs/datasets/${DATASET}.yaml \
    --config-file ${DIR_BASE}/configs/trainers/${CONFIG_BASE}/${CONFIG_NAME}.yaml \
    --output-dir  ${eval_output_dir} \
    --model-dir ${train_output_dir} \
    --ood_method mcm \
    --load-epoch 50 \
    --eval-only \
    --eval_full_supectrum_ood \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
    DATASET.ID_CLASSES_FILE "${DATA}/class_splits/imagenet/${CUSTOM_SPLIT_MODE}/id_data.txt" \
    DATASET.OOD_CLASSES_FILE "${DATA}/class_splits/imagenet/${CUSTOM_SPLIT_MODE}/ood_data.txt"

done