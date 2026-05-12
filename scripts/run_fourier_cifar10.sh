#!/bin/bash

# FINER CIFAR-10 pipeline:
#   1. meta-train a ModulatedFINER backbone,
#   2. create a 5000-train / 1000-test functaset,
#   3. combine train+val back into 5000 train samples,
#   4. train downstream parameter-space classifier.
#
# No reconstruction eval.
# No PGD attack.

set -euo pipefail

# ---- architecture ------------------------------------------------------------

HIDDEN_DIM=512
MOD_DIM=512
DEPTH=10

INR_TYPE=finer
COORD_TAG="norm01"

FINER_FREQ=30.0
FINER_FIRST_BIAS_SCALE=2.0
FINER_SCALE_REQ_GRAD=0

# ---- meta-training hyperparameters ------------------------------------------

EPOCHS=5
INT_LR=0.01
INNER_STEPS=3

EXT_LR=1e-5
TRAIN_BATCH_SIZE=32

CUDA_GPU=1
LOG_SIGMAS_EVERY=50

# ---- functaset creation hyperparameters -------------------------------------

MAKESET_ITERS=200
MAKESET_INNER_OPTIM=adam
MAKESET_LR=0.003

MAX_TRAIN_SAMPLES=5000
MAX_TEST_SAMPLES=1000

# ---- classifier hyperparameters ---------------------------------------------

CLF_LR=0.001
CLF_WIDTH=512
CLF_DEPTH=2
CLF_DROPOUT=0.5
CLF_BATCH_SIZE=256
CLF_EPOCHS=120

# -----------------------------------------------------------------------------

EXT_LR_TAG=$(printf "%.0e" "${EXT_LR}")
BIAS_TAG=$(printf "%g" "${FINER_FIRST_BIAS_SCALE}" | tr '.' 'p')
MAKESET_LR_TAG="3e-03"

if [[ "${FINER_SCALE_REQ_GRAD}" -eq 1 ]]; then
    SCALE_TAG="scalegrad"
else
    SCALE_TAG="scaledetach"
fi

SLUG="functa_like_cifar10_finer_h${HIDDEN_DIM}_md${MOD_DIM}_d${DEPTH}_freq${FINER_FREQ}_bias${BIAS_TAG}_${SCALE_TAG}_${COORD_TAG}_extlr${EXT_LR_TAG}_e${EPOCHS}_inner${INNER_STEPS}_adamphi${MAKESET_ITERS}_lr${MAKESET_LR_TAG}_train${MAX_TRAIN_SAMPLES}_test${MAX_TEST_SAMPLES}"

VARIANT_FLAGS=(
    --variant vanilla
)

INR_FLAGS=(
    --inr-type "${INR_TYPE}"
    --finer-freq "${FINER_FREQ}"
    --finer-first-bias-scale "${FINER_FIRST_BIAS_SCALE}"
)

if [[ "${FINER_SCALE_REQ_GRAD}" -eq 1 ]]; then
    INR_FLAGS+=(--finer-scale-req-grad)
fi

MODEL_DIR="model_cifar10/${SLUG}"
CHECKPOINT="${MODEL_DIR}/modSiren.pth"
RUN_ROOT="runs/${SLUG}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "============================================================"
echo "FINER CIFAR-10 subset pipeline"
echo "dataset       = cifar10"
echo "inr_type      = ${INR_TYPE}"
echo "hidden_dim    = ${HIDDEN_DIM}"
echo "mod_dim       = ${MOD_DIM}"
echo "depth         = ${DEPTH}"
echo "finer freq    = ${FINER_FREQ}"
echo "finer bias    = ${FINER_FIRST_BIAS_SCALE}"
echo "scale grad    = ${FINER_SCALE_REQ_GRAD}"
echo "meta epochs   = ${EPOCHS}"
echo "meta int_lr   = ${INT_LR}"
echo "meta ext_lr   = ${EXT_LR}"
echo "inner steps   = ${INNER_STEPS}"
echo "make iters    = ${MAKESET_ITERS}"
echo "make optim    = ${MAKESET_INNER_OPTIM}"
echo "make lr       = ${MAKESET_LR}"
echo "train samples = ${MAX_TRAIN_SAMPLES}"
echo "test samples  = ${MAX_TEST_SAMPLES}"
echo "slug          = ${SLUG}"
echo "checkpoint    = ${CHECKPOINT}"
echo "run root      = ${RUN_ROOT}"
echo "GPU           = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/3: Training FINER CIFAR-10 backbone"
echo "          -> ${CHECKPOINT}"
echo

python trainer.py \
    --dataset cifar10 \
    --data-path ../data \
    --device cuda \
    --num-epochs "${EPOCHS}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --int-lr "${INT_LR}" \
    --ext-lr "${EXT_LR}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --mod-dim "${MOD_DIM}" \
    --depth "${DEPTH}" \
    --inner-steps "${INNER_STEPS}" \
    "${INR_FLAGS[@]}" \
    --model-name "${SLUG}" \
    "${VARIANT_FLAGS[@]}" \
    --log-sigmas-every "${LOG_SIGMAS_EVERY}"

echo
echo "Verifying saved checkpoint"
echo "          -> ${CHECKPOINT}"
echo

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: expected checkpoint was not created: ${CHECKPOINT}" >&2
    exit 1
fi

python - <<PYCKPT
import sys
import torch

ckpt_path = "${CHECKPOINT}"
expected_model_name = "${SLUG}"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("checkpoint metadata:")
print("  epoch:", ckpt.get("epoch"))
print("  loss:", ckpt.get("loss"))
print("  variant:", ckpt.get("variant"))
print("  model_name:", ckpt.get("model_name"))
print("  model_args:", ckpt.get("model_args"))

if ckpt.get("model_name") != expected_model_name:
    print(
        f"ERROR: checkpoint model_name={ckpt.get('model_name')!r} "
        f"does not match expected {expected_model_name!r}",
        file=sys.stderr,
    )
    sys.exit(1)

model_args = ckpt.get("model_args", {})

checks = {
    "dataset": "cifar10",
    "inr_type": "finer",
    "hidden_dim": ${HIDDEN_DIM},
    "mod_dim": ${MOD_DIM},
    "depth": ${DEPTH},
    "finer_first_bias_scale": ${FINER_FIRST_BIAS_SCALE},
}

for key, expected in checks.items():
    got = model_args.get(key)
    if got != expected:
        print(f"ERROR: model_args[{key!r}]={got!r}, expected {expected!r}", file=sys.stderr)
        sys.exit(1)

print("Checkpoint verified.")
PYCKPT

echo
echo "Step 2/3: Creating 5000-train / 1000-test CIFAR-10 functaset"
echo "          -> ${RUN_ROOT}/functaset/${SLUG}_{train,val,test}.pkl"
echo

rm -rf "${RUN_ROOT}/functaset"
mkdir -p "${RUN_ROOT}"

python makeset.py \
    --dataset cifar10 \
    --data-path ../data \
    --iters "${MAKESET_ITERS}" \
    --lr "${MAKESET_LR}" \
    --inner-optim "${MAKESET_INNER_OPTIM}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-test-samples "${MAX_TEST_SAMPLES}" \
    --checkpoint "${CHECKPOINT}" \
    --saveroot "${RUN_ROOT}" \
    --device cuda \
    --functaset-stem "${SLUG}" \
    "${INR_FLAGS[@]}" \
    "${VARIANT_FLAGS[@]}"

echo
echo "Combining train + val into one ${MAX_TRAIN_SAMPLES}-sample train set"
echo

python - <<PYCOMBINE
import joblib
from collections import Counter

root = "${RUN_ROOT}/functaset"
slug = "${SLUG}"

train = joblib.load(f"{root}/{slug}_train.pkl")
val = joblib.load(f"{root}/{slug}_val.pkl")
test = joblib.load(f"{root}/{slug}_test.pkl")

combined = train + val

out_path = f"{root}/{slug}_train_all${MAX_TRAIN_SAMPLES}.pkl"
joblib.dump(combined, out_path)

print("train:", len(train), Counter([x["label"] for x in train]))
print("val:", len(val), Counter([x["label"] for x in val]))
print("combined:", len(combined), Counter([x["label"] for x in combined]))
print("test:", len(test), Counter([x["label"] for x in test]))
print("saved:", out_path)
PYCOMBINE

echo
echo "Step 3/3: Training downstream CIFAR-10 classifier"
echo "          -> ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo

pushd "${RUN_ROOT}" > /dev/null

python ~/SIREN_Vista/train_classifier.py \
    --lr "${CLF_LR}" \
    --cwidth "${CLF_WIDTH}" \
    --mod-dim "${MOD_DIM}" \
    --dropout "${CLF_DROPOUT}" \
    --cdepth "${CLF_DEPTH}" \
    --batch-size "${CLF_BATCH_SIZE}" \
    --dataset cifar10 \
    --num-epochs "${CLF_EPOCHS}" \
    --data-path ~/data \
    --functaset-path-train "./functaset/${SLUG}_train_all${MAX_TRAIN_SAMPLES}.pkl" \
    --functaset-path-test "./functaset/${SLUG}_test.pkl" \
    --device cuda

popd > /dev/null

echo
echo "============================================================"
echo "Done."
echo "FINER checkpoint : ${CHECKPOINT}"
echo "Functaset train  : ${RUN_ROOT}/functaset/${SLUG}_train_all${MAX_TRAIN_SAMPLES}.pkl"
echo "Functaset test   : ${RUN_ROOT}/functaset/${SLUG}_test.pkl"
echo "Classifier       : ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo "============================================================"