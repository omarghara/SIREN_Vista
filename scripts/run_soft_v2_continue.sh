#!/bin/bash

# Continue hardcap90 soft-Lipschitz pipeline from existing checkpoint:
#   2. create functaset
#   3. train downstream classifier
#   4. evaluate reconstruction quality
#
# This script does NOT train the SIREN backbone and does NOT run PGD.

set -euo pipefail

# ---- hyperparameters ---------------------------------------------------------

# Existing checkpoint from your completed hardcap90 run.
CHECKPOINT="model_mnist/softlip_L1_lam1e+00_sine_and_readout/modSiren.pth"

# We choose the run root manually so outputs are clearly named as hardcap90.
RUN_ROOT="runs/softlip_hardcap90_lam1e+00_sine_and_readout"

# Variant flags must match the checkpoint metadata.
L=1.0
LAM=1
APPLY_TO=sine_and_readout

# Modulation fitting.
MAKESET_ITERS=5
INT_LR=0.01

# GPU.
CUDA_GPU=1

# Reconstruction evaluation.
EVAL_ITERS="5,20,50,100,200"
EVAL_MAX_SAMPLES=2000
EVAL_BATCH_SIZE=128

# -----------------------------------------------------------------------------

VARIANT_FLAGS=(
    --variant soft_lipschitz
    --soft-lip-cap "${L}"
    --soft-lip-lambda "${LAM}"
    --soft-lip-apply-to "${APPLY_TO}"
)

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "== Continue hardcap90 soft-Lipschitz pipeline: steps 2, 3, 4 =="
echo "   checkpoint  = ${CHECKPOINT}"
echo "   run root    = ${RUN_ROOT}"
echo "   L           = ${L}"
echo "   lambda      = ${LAM}"
echo "   apply_to    = ${APPLY_TO}"
echo "   make iters  = ${MAKESET_ITERS}"
echo "   CUDA device = ${CUDA_VISIBLE_DEVICES}"
echo

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found:"
    echo "  ${CHECKPOINT}"
    exit 1
fi

python - <<PY
import torch
ckpt_path = "${CHECKPOINT}"
ckpt = torch.load(ckpt_path, map_location="cpu")
print("checkpoint metadata:")
print("  epoch:", ckpt.get("epoch"))
print("  loss:", ckpt.get("loss"))
print("  variant:", ckpt.get("variant"))
print("  variant_args:", ckpt.get("variant_args"))
PY

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 2/4: Creating functaset"
echo "          -> ${RUN_ROOT}/functaset/mnist_{train,val,test}.pkl"

rm -rf "${RUN_ROOT}/functaset"
mkdir -p "${RUN_ROOT}"

python makeset.py \
    --dataset mnist \
    --data-path ../data \
    --iters "${MAKESET_ITERS}" \
    --checkpoint "${CHECKPOINT}" \
    --saveroot "${RUN_ROOT}" \
    --device cuda \
    "${VARIANT_FLAGS[@]}"

echo
echo "Step 3/4: Training downstream classifier"
echo "          -> ${RUN_ROOT}/mnist_classifier/best_classifier.pth"

pushd "${RUN_ROOT}" > /dev/null

python ~/SIREN_Vista/train_classifier.py \
    --lr 0.01 \
    --cwidth 512 \
    --mod-dim 512 \
    --dropout 0.2 \
    --cdepth 3 \
    --batch-size 256 \
    --dataset mnist \
    --num-epochs 40 \
    --data-path ~/data \
    --functaset-path-train ./functaset/mnist_train.pkl \
    --functaset-path-test ./functaset/mnist_test.pkl \
    --device cuda

popd > /dev/null

echo
echo "Step 4/4: Evaluating reconstruction quality (MSE / PSNR / SSIM)"
echo "          -> ${RUN_ROOT}/reconstruction_eval.json"

EVAL_CAP_ARGS=()
if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
    EVAL_CAP_ARGS=(--max-samples "${EVAL_MAX_SAMPLES}")
fi

python evaluate_reconstruction.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset mnist \
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
echo "Done."
echo "Checkpoint      : ${CHECKPOINT}"
echo "Functaset       : ${RUN_ROOT}/functaset/mnist_{train,val,test}.pkl"
echo "Classifier      : ${RUN_ROOT}/mnist_classifier/best_classifier.pth"
echo "Recon eval JSON : ${RUN_ROOT}/reconstruction_eval.json"