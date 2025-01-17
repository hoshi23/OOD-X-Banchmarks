DIR_BASE="../.."
DATA="${DIR_BASE}/data"
OUTPUT_BASE="${DIR_BASE}/output"

# config
CONFIG_BASE="CoOp"
CONFIG_NAME="vit_b16"
TRAINER="ZeroshotCLIPLocal"
SHOTS=1 # This is to reduce the time required for loading data.
SEED=1
TRAIN_MODE="MCM"

# data mode
DATASET="imagenet_x"
SUBSAMPLE_CLASSES="custom"
CUSTOM_SPLIT_MODE="x"


# output
ADDITIONAL_SETTING="${SUBSAMPLE_CLASSES}_${CUSTOM_SPLIT_MODE}"
eval_output_dir="${OUTPUT_BASE}/${TRAINER}/shots_${SHOTS}/${CONFIG_NAME}/${TRAIN_MODE}/${DATASET}/${ADDITIONAL_SETTING}/eval/seed${SEED}"




# ImageNet-X
echo "Benchmark: ImageNet-X"
python ${DIR_BASE}/src/train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file ${DIR_BASE}/configs/datasets/${DATASET}.yaml \
--config-file ${DIR_BASE}/configs/trainers/${CONFIG_BASE}/${CONFIG_NAME}.yaml \
--output-dir  ${eval_output_dir} \
--ood_method mcm glmcm \
--eval-only \
--eval_full_supectrum_ood \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.PROMPT MCM \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATASET.ID_CLASSES_FILE "${DATA}/class_splits/imagenet/${CUSTOM_SPLIT_MODE}/first_datasets.txt" \
DATASET.OOD_CLASSES_FILE "${DATA}/class_splits/imagenet/${CUSTOM_SPLIT_MODE}/second_datasets.txt"
