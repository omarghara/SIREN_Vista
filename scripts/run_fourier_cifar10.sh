#!/bin/bash

# Functa-like FINER CIFAR-10 pipeline:
#   1. meta-train a ModulatedFINER backbone,
#   2. create a CIFAR-10 functaset,
#   3. train downstream parameter-space classifier,
#   4. evaluate reconstruction quality,
#   5. skip PGD for now.

set -euo pipefail

# ---- architecture ------------------------------------------------------------

HIDDEN_DIM=512
MOD_DIM=1024
DEPTH=15

INR_TYPE=finer
COORD_TAG="norm01"

FINER_FREQ=30.0
FINER_FIRST_BIAS_SCALE=2.0
FINER_SCALE_REQ_GRAD=0

# ---- training hyperparameters ------------------------------------------------

EPOCHS=10
INT_LR=0.01
INNER_STEPS=3

EXT_LR=1e-5
TRAIN_BATCH_SIZE=32

MAKESET_ITERS=50
MAKESET_INNER_OPTIM=sgd

CUDA_GPU=1
LOG_SIGMAS_EVERY=50

EVAL_ITERS="5,10,20,50,100,200,500"
EVAL_MAX_SAMPLES=2000
EVAL_BATCH_SIZE=64
EVAL_INNER_OPTIM=sgd

RUN_PGD=0

# ---- classifier hyperparameters ---------------------------------------------

CLF_LR=0.01
CLF_WIDTH=1024
CLF_DEPTH=4
CLF_DROPOUT=0.3
CLF_BATCH_SIZE=256
CLF_EPOCHS=80

# -----------------------------------------------------------------------------

EXT_LR_TAG=$(printf "%.0e" "${EXT_LR}")
BIAS_TAG=$(printf "%g" "${FINER_FIRST_BIAS_SCALE}" | tr '.' 'p')

if [[ "${FINER_SCALE_REQ_GRAD}" -eq 1 ]]; then
    SCALE_TAG="scalegrad"
else
    SCALE_TAG="scaledetach"
fi

SLUG="functa_like_cifar10_finer_h${HIDDEN_DIM}_md${MOD_DIM}_d${DEPTH}_freq${FINER_FREQ}_bias${BIAS_TAG}_${SCALE_TAG}_${COORD_TAG}_extlr${EXT_LR_TAG}_e${EPOCHS}_inner${INNER_STEPS}_make${MAKESET_ITERS}"

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

echo "== Functa-like FINER CIFAR-10 pipeline =="
echo "   dataset       = cifar10"
echo "   inr_type      = ${INR_TYPE}"
echo "   hidden_dim    = ${HIDDEN_DIM}"
echo "   mod_dim       = ${MOD_DIM}"
echo "   depth         = ${DEPTH}"
echo "   coord norm    = ${COORD_TAG} pixel centers"
echo "   finer freq    = ${FINER_FREQ}"
echo "   finer bias    = ${FINER_FIRST_BIAS_SCALE}"
echo "   scale grad    = ${FINER_SCALE_REQ_GRAD}"
echo "   int_lr        = ${INT_LR}"
echo "   inner steps   = ${INNER_STEPS}"
echo "   ext_lr        = ${EXT_LR}"
echo "   epochs        = ${EPOCHS}"
echo "   train batch   = ${TRAIN_BATCH_SIZE}"
echo "   make iters    = ${MAKESET_ITERS}"
echo "   make optim    = ${MAKESET_INNER_OPTIM}"
echo "   slug          = ${SLUG}"
echo "   checkpoint    = ${CHECKPOINT}"
echo "   run root      = ${RUN_ROOT}"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/5: Training FINER CIFAR-10 backbone"
echo "          -> ${CHECKPOINT}"

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
if model_args.get("inr_type") != "finer":
    print("ERROR: checkpoint is not FINER:", model_args.get("inr_type"), file=sys.stderr)
    sys.exit(1)
PYCKPT

echo
echo "Step 2/5: Creating CIFAR-10 functaset"
echo "          -> ${RUN_ROOT}/functaset/${SLUG}_{train,val,test}.pkl"

rm -rf "${RUN_ROOT}/functaset"
mkdir -p "${RUN_ROOT}"

python makeset.py \
    --dataset cifar10 \
    --data-path ../data \
    --iters "${MAKESET_ITERS}" \
    --lr "${INT_LR}" \
    --inner-optim "${MAKESET_INNER_OPTIM}" \
    --checkpoint "${CHECKPOINT}" \
    --saveroot "${RUN_ROOT}" \
    --device cuda \
    --functaset-stem "${SLUG}" \
    "${INR_FLAGS[@]}" \
    "${VARIANT_FLAGS[@]}"

echo
echo "Step 3/5: Training downstream CIFAR-10 classifier"
echo "          -> ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"

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
    --functaset-path-train "./functaset/${SLUG}_train.pkl" \
    --functaset-path-test "./functaset/${SLUG}_test.pkl" \
    --device cuda

popd > /dev/null

echo
echo "Step 4/5: Evaluating CIFAR-10 reconstruction quality"
echo "          -> ${RUN_ROOT}/reconstruction_eval.json"

EVAL_CAP_ARGS=()
if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
    EVAL_CAP_ARGS=(--max-samples "${EVAL_MAX_SAMPLES}")
fi

python evaluate_reconstruction.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset cifar10 \
    --data-path ../data \
    --device cuda \
    "${INR_FLAGS[@]}" \
    "${VARIANT_FLAGS[@]}" \
    --iter-checkpoints "${EVAL_ITERS}" \
    --split both \
    --inner-lr "${INT_LR}" \
    --inner-optim "${EVAL_INNER_OPTIM}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --output "${RUN_ROOT}/reconstruction_eval.json" \
    "${EVAL_CAP_ARGS[@]}"

echo
echo "Step 5/5: Full-PGD adversarial attack"
echo "          skipped because RUN_PGD=0"

echo
echo "Done."
echo "SIREN/FINER checkpoint : ${CHECKPOINT}"
echo "Functaset              : ${RUN_ROOT}/functaset/${SLUG}_{train,val,test}.pkl"
echo "Classifier             : ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo "Recon eval JSON        : ${RUN_ROOT}/reconstruction_eval.json"