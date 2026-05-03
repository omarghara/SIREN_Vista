#!/bin/bash

# Vanilla CIFAR-10 pipeline:
#   1. meta-train a vanilla CIFAR-10 SIREN backbone,
#   2. create a CIFAR-10 functaset,
#   3. train the downstream parameter-space classifier,
#   4. evaluate reconstruction quality,
#   5. skip PGD for now.
#
# CIFAR-10 uses a 32x32 coordinate grid and a 3-channel readout.

set -euo pipefail

# ---- hyperparameters ---------------------------------------------------------

EPOCHS=5
INT_LR=0.01
EXT_LR=5e-5
TRAIN_BATCH_SIZE=128

MAKESET_ITERS=20

CUDA_GPU=1
LOG_SIGMAS_EVERY=50

EVAL_ITERS="5,20,50,100,200"
EVAL_MAX_SAMPLES=2000
EVAL_BATCH_SIZE=64

RUN_PGD=0

# -----------------------------------------------------------------------------

SLUG="vanilla_cifar10"

VARIANT_FLAGS=(
    --variant vanilla
)

MODEL_DIR="model_cifar10/${SLUG}"
CHECKPOINT="${MODEL_DIR}/modSiren.pth"
RUN_ROOT="runs/${SLUG}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "== vanilla CIFAR-10 pipeline =="
echo "   dataset       = cifar10"
echo "   variant       = vanilla"
echo "   int_lr        = ${INT_LR}"
echo "   ext_lr        = ${EXT_LR}"
echo "   epochs        = ${EPOCHS}"
echo "   train batch   = ${TRAIN_BATCH_SIZE}"
echo "   make iters    = ${MAKESET_ITERS}"
echo "   slug          = ${SLUG}"
echo "   CUDA device   = ${CUDA_VISIBLE_DEVICES}"
echo "   checkpoint    = ${CHECKPOINT}"
echo "   run root      = ${RUN_ROOT}"
echo "   PGD enabled   = ${RUN_PGD}"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/5: Training vanilla CIFAR-10 SIREN backbone"
echo "          -> ${CHECKPOINT}"

python trainer.py \
    --dataset cifar10 \
    --data-path ../data \
    --device cuda \
    --num-epochs "${EPOCHS}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --int-lr "${INT_LR}" \
    --ext-lr "${EXT_LR}" \
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
PYCKPT

echo
echo "Step 2/5: Creating CIFAR-10 functaset"
echo "          -> ${RUN_ROOT}/functaset/cifar10_{train,val,test}.pkl"

rm -rf "${RUN_ROOT}/functaset"
mkdir -p "${RUN_ROOT}"

python makeset.py \
    --dataset cifar10 \
    --data-path ../data \
    --iters "${MAKESET_ITERS}" \
    --checkpoint "${CHECKPOINT}" \
    --saveroot "${RUN_ROOT}" \
    --device cuda \
    "${VARIANT_FLAGS[@]}"

echo
echo "Step 3/5: Training downstream CIFAR-10 classifier"
echo "          -> ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"

pushd "${RUN_ROOT}" > /dev/null

python ~/SIREN_Vista/train_classifier.py \
    --lr 0.01 \
    --cwidth 512 \
    --mod-dim 512 \
    --dropout 0.2 \
    --cdepth 3 \
    --batch-size 256 \
    --dataset cifar10 \
    --num-epochs 40 \
    --data-path ~/data \
    --functaset-path-train ./functaset/cifar10_train.pkl \
    --functaset-path-test ./functaset/cifar10_test.pkl \
    --device cuda

popd > /dev/null

echo
echo "Step 4/5: Evaluating CIFAR-10 reconstruction quality (MSE / PSNR / SSIM)"
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
    "${VARIANT_FLAGS[@]}" \
    --iter-checkpoints "${EVAL_ITERS}" \
    --split both \
    --inner-lr "${INT_LR}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --output "${RUN_ROOT}/reconstruction_eval.json" \
    "${EVAL_CAP_ARGS[@]}"

echo
echo "Step 5/5: Full-PGD adversarial attack"

if [[ "${RUN_PGD}" -eq 1 ]]; then
    echo "ERROR: CIFAR-10 PGD is not wired in attacks/full_pgd.py yet." >&2
    exit 1
else
    echo "          skipped because RUN_PGD=0"
    PGD_LOG="${RUN_ROOT}/pgd_attack.log"
fi

echo
echo "Done."
echo "SIREN checkpoint : ${CHECKPOINT}"
echo "Functaset        : ${RUN_ROOT}/functaset/cifar10_{train,val,test}.pkl"
echo "Classifier       : ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo "Recon eval JSON  : ${RUN_ROOT}/reconstruction_eval.json"
echo "PGD attack log   : skipped"
