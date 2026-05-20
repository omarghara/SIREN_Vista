#!/bin/bash

# Hardcoded per-layer soft-Lipschitz MNIST pipeline:
#   1. meta-train the SIREN backbone with hardcoded 90%-of-vanilla spectral caps,
#   2. fit per-sample modulations into a functaset,
#   3. train the downstream parameter-space classifier on that functaset,
#   4. evaluate reconstruction quality (MSE / PSNR / SSIM) vs inner-loop iter count,
#   5. skip Full-PGD for now.
#
# Experiment goal:
#   Reduce each SIREN layer spectral norm by about 10% relative to vanilla,
#   while NOT penalizing the modulation matrix.
#
# Penalized:
#   - sine.0
#   - sine.1 ... sine.9
#   - readout / hidden2rgb
#
# Not penalized:
#   - modul
#
# Note:
#   L is kept only because the CLI still expects --soft-lip-cap.
#   The hardcoded _collect_layers() ignores L for this experiment.

set -euo pipefail

# ---- hyperparameters ---------------------------------------------------------

# For this hardcoded-cap experiment, L is not actually used by _collect_layers.
# Keep it only because --soft-lip-cap is still part of the CLI.
L=1.0

# Penalty strength.
LAM=1

# We want only SIREN weights:
#   sine layers + readout
# Do NOT use "all" because "all" conceptually includes modul.
APPLY_TO=sine_and_readout

# Hardcoded caps already include sine.0.
# Keep SKIP_FIRST=0 for clarity.
SKIP_FIRST=0

# Meta-training.
EPOCHS=6
INT_LR=0.01
EXT_LR=5e-5

# Fitting modulations for functaset.
MAKESET_ITERS=5

# GPU.
CUDA_GPU=1

# Logging spectral norms during training.
LOG_SIGMAS_EVERY=50

# Reconstruction evaluation.
EVAL_ITERS="5,20,50,100,200"
EVAL_MAX_SAMPLES=2000
EVAL_BATCH_SIZE=128

# PGD disabled for this experiment.
RUN_PGD=0

# PGD settings kept here for later, but not used when RUN_PGD=0.
PGD_EPSILON=16
PGD_STEPS=100
PGD_MOD_STEPS=10
PGD_LR=0.01
PGD_INNER_LR=0.01

# -----------------------------------------------------------------------------

# Slug for this hardcoded cap experiment.
LAM_TAG=$(printf "%.0e" "$LAM")
SLUG="softlip_hardcap90_lam${LAM_TAG}_${APPLY_TO}"

# Variant flags passed to trainer / makeset / reconstruction eval.
VARIANT_FLAGS=(
    --variant soft_lipschitz
    --soft-lip-cap "${L}"
    --soft-lip-lambda "${LAM}"
    --soft-lip-apply-to "${APPLY_TO}"
)

if [[ "${SKIP_FIRST}" -eq 1 ]]; then
    VARIANT_FLAGS+=(--soft-lip-skip-first)
fi

MODEL_DIR="model_mnist/${SLUG}"
CHECKPOINT="${MODEL_DIR}/modSiren.pth"
RUN_ROOT="runs/${SLUG}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "== hardcoded per-layer soft-Lipschitz MNIST pipeline =="
echo "   experiment    = hardcap90"
echo "   meaning       = per-layer sigma caps are 90% of vanilla values"
echo "   L            = ${L}   (kept for CLI; ignored by hardcoded _collect_layers)"
echo "   lambda       = ${LAM}"
echo "   apply_to     = ${APPLY_TO}"
echo "   skip_first   = ${SKIP_FIRST}"
echo "   int_lr       = ${INT_LR}"
echo "   ext_lr       = ${EXT_LR}"
echo "   epochs       = ${EPOCHS}"
echo "   make iters   = ${MAKESET_ITERS}"
echo "   slug         = ${SLUG}"
echo "   CUDA device  = ${CUDA_VISIBLE_DEVICES}"
echo "   checkpoint   = ${CHECKPOINT}"
echo "   run root     = ${RUN_ROOT}"
echo "   PGD enabled  = ${RUN_PGD}"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/5: Training hardcoded-cap soft-Lipschitz SIREN backbone"
echo "          -> ${CHECKPOINT}"

python trainer.py \
    --dataset mnist \
    --data-path ../data \
    --device cuda \
    --num-epochs "${EPOCHS}" \
    --model-name "${SLUG}" \
    --int-lr "${INT_LR}" \
    --ext-lr "${EXT_LR}" \
    "${VARIANT_FLAGS[@]}" \
    --log-sigmas-every "${LOG_SIGMAS_EVERY}"

echo
echo "Step 2/5: Creating functaset"
echo "          -> ${RUN_ROOT}/functaset/mnist_{train,val,test}.pkl"

# makeset's split() calls os.makedirs without exist_ok, so wipe any partial run.
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
echo "Step 3/5: Training downstream classifier"
echo "          -> ${RUN_ROOT}/mnist_classifier/best_classifier.pth"

# Run from RUN_ROOT so train_classifier's default mnist_classifier/ lands
# inside the variant's run dir instead of at SIREN_Vista root.
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
echo "Step 4/5: Evaluating reconstruction quality (MSE / PSNR / SSIM)"
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
echo "Step 5/5: Full-PGD adversarial attack"

if [[ "${RUN_PGD}" -eq 1 ]]; then
    echo "          -> ${RUN_ROOT}/pgd_attack.log"

    CLASSIFIER_CKPT="${RUN_ROOT}/mnist_classifier/best_classifier.pth"
    PGD_LOG="${RUN_ROOT}/pgd_attack.log"

    python -u attacks/full_pgd.py \
        --dataset mnist \
        --data-path ../data \
        --siren-checkpoint "${CHECKPOINT}" \
        --classifier-checkpoint "${CLASSIFIER_CKPT}" \
        --epsilon "${PGD_EPSILON}" \
        --pgd-steps "${PGD_STEPS}" \
        --mod-steps "${PGD_MOD_STEPS}" \
        --ext-lr "${PGD_LR}" \
        --inner-lr "${PGD_INNER_LR}" \
        --cwidth 512 \
        --cdepth 3 \
        --mod-dim 512 \
        --hidden-dim 256 \
        --depth 10 \
        --device cuda 2>&1 | tee "${PGD_LOG}"
else
    echo "          skipped because RUN_PGD=0"
    PGD_LOG="${RUN_ROOT}/pgd_attack.log"
fi

echo
echo "Done."
echo "SIREN checkpoint : ${CHECKPOINT}"
echo "Functaset        : ${RUN_ROOT}/functaset/mnist_{train,val,test}.pkl"
echo "Classifier       : ${RUN_ROOT}/mnist_classifier/best_classifier.pth"
echo "Recon eval JSON  : ${RUN_ROOT}/reconstruction_eval.json"

if [[ "${RUN_PGD}" -eq 1 ]]; then
    echo "PGD attack log   : ${PGD_LOG}"
else
    echo "PGD attack log   : skipped"
fi