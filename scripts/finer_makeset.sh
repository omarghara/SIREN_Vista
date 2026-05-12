#!/bin/bash
set -euo pipefail

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES=1

cd ~/SIREN_Vista || exit 1

BASE_SLUG="functa_like_cifar10_finer_h512_md1024_d15_freq30.0_bias2_scaledetach_norm01_extlr1e-05_e10_inner3_make50"
CHECKPOINT="model_cifar10/${BASE_SLUG}/modSiren.pth"

MAKE_ITERS=200
INNER_OPTIM="adam"
INNER_LR=0.003

MAX_TRAIN_SAMPLES=5000
MAX_TEST_SAMPLES=1000

NEW_SLUG="${BASE_SLUG}_adamphi${MAKE_ITERS}_lr3e-03_train${MAX_TRAIN_SAMPLES}_test${MAX_TEST_SAMPLES}"
RUN_ROOT="runs/${NEW_SLUG}"

echo "============================================================"
echo "Creating CIFAR-10 FINER functaset subset only"
echo "Checkpoint : ${CHECKPOINT}"
echo "Output root: ${RUN_ROOT}"
echo "Stem       : ${NEW_SLUG}"
echo "Optimizer  : ${INNER_OPTIM}"
echo "Inner iters: ${MAKE_ITERS}"
echo "Inner lr   : ${INNER_LR}"
echo "Train n    : ${MAX_TRAIN_SAMPLES}"
echo "Test n     : ${MAX_TEST_SAMPLES}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint does not exist:"
    echo "  ${CHECKPOINT}"
    exit 1
fi

mkdir -p "${RUN_ROOT}"

python - <<PYCKPT
import torch
ckpt_path = "${CHECKPOINT}"
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Checkpoint metadata:")
print("  epoch:", ckpt.get("epoch"))
print("  loss:", ckpt.get("loss"))
print("  variant:", ckpt.get("variant"))
print("  model_name:", ckpt.get("model_name"))
print("  model_args:", ckpt.get("model_args"))
model_args = ckpt.get("model_args", {})
assert model_args.get("dataset") == "cifar10", model_args
assert model_args.get("inr_type") == "finer", model_args
assert model_args.get("finer_first_bias_scale") == 2.0, model_args
print("Checkpoint verified: CIFAR-10 FINER bias_scale=2.0")
PYCKPT

echo
echo "Running makeset.py..."
echo

python makeset.py \
    --dataset cifar10 \
    --data-path ../data \
    --checkpoint "${CHECKPOINT}" \
    --saveroot "${RUN_ROOT}" \
    --functaset-stem "${NEW_SLUG}" \
    --iters "${MAKE_ITERS}" \
    --lr "${INNER_LR}" \
    --inner-optim "${INNER_OPTIM}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-test-samples "${MAX_TEST_SAMPLES}" \
    --device cuda \
    --variant vanilla

echo
echo "============================================================"
echo "Done creating functaset subset."
echo
echo "Train:"
echo "  ${RUN_ROOT}/functaset/${NEW_SLUG}_train.pkl"
echo "Val:"
echo "  ${RUN_ROOT}/functaset/${NEW_SLUG}_val.pkl"
echo "Test:"
echo "  ${RUN_ROOT}/functaset/${NEW_SLUG}_test.pkl"
echo "============================================================"