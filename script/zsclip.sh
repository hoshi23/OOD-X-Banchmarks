DIR_BASE=".."


# config
CONFIG_BASE="CoOp"
CONFIG_NAME="vit_b16"
TRAINER="ZeroshotCLIPLocal"

# data mode
DATA="${DIR_BASE}/data"
DATASET="imagenet_x"
SUBSAMPLE_CLASSES="custom"
CUSTOM_SPLIT_MODE="subclass_base_hard"
SHOTS=1 # This is to reduce the time required for loading data.
SEED=1

# train mode
TRAIN_MODE="MCM"
TRAIN_SETTING="${SUBSAMPLE_CLASSES}_${CUSTOM_SPLIT_MODE}"

# output
OUTPUT_BASE="${DIR_BASE}/output"
eval_output_dir="${OUTPUT_BASE}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CONFIG_NAME}/${TRAIN_MODE}/${TRAIN_SETTING}/eval/seed${SEED}"


echo "SEED: $SEED"
echo "Dataset: $DATASET"


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
DATASET.ID_CLASSES_FILE "${DATA}/class_splits/imagenet/x/first_datasets.txt" \
DATASET.OOD_CLASSES_FILE "${DATA}/class_splits/imagenet/x/second_datasets.txt"
