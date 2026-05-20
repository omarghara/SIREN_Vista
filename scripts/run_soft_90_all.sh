#!/bin/bash

# First95-Rest80 soft-Lipschitz MNIST pipeline:
#   1. meta-train SIREN with:
#        - sine.0 cap       = 95% of vanilla sigma
#        - sine.1-9 caps    = 80% of vanilla sigma
#        - readout cap      = 80% of vanilla sigma
#        - modul is NOT capped
#   2. create functaset
#   3. train classifier
#   4. evaluate reconstruction
#   5. skip PGD for now
#
# Important:
#   This assumes variants/soft_lipschitz.py has the hardcoded
#   first95_rest80 _collect_layers() implementation:
#       sine.0       = 0.95 * vanilla sigma
#       sine.1-9     = 0.80 * vanilla sigma
#       readout      = 0.80 * vanilla sigma
#       no modul cap
#
#   Use APPLY_TO=sine_and_readout so only SIREN/readout weights are penalized.

set -euo pipefail

# ---- hyperparameters ---------------------------------------------------------

# Kept for CLI compatibility. The hardcoded _collect_layers ignores this
# for actual caps.
L=1.0

LAM=1
APPLY_TO=sine_and_readout
SKIP_FIRST=0

EPOCHS=5
INT_LR=0.01
EXT_LR=5e-5

MAKESET_ITERS=5

CUDA_GPU=1
LOG_SIGMAS_EVERY=50

EVAL_ITERS="5,20,50,100,200"
EVAL_MAX_SAMPLES=2000
EVAL_BATCH_SIZE=128

RUN_PGD=0

PGD_EPSILON=16
PGD_STEPS=100
PGD_MOD_STEPS=10
PGD_LR=0.01
PGD_INNER_LR=0.01

# -----------------------------------------------------------------------------

LAM_TAG=$(printf "%.0e" "$LAM")
SLUG="softlip_first95_rest80_lam${LAM_TAG}_${APPLY_TO}"

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

echo "== first95_rest80 soft-Lipschitz MNIST pipeline =="
echo "   experiment    = first95_rest80"
echo "   meaning       = sine.0 cap is 95% of vanilla; sine.1-9/readout caps are 80%; modul is not capped"
echo "   L             = ${L}   (kept for CLI; ignored by hardcoded _collect_layers)"
echo "   lambda        = ${LAM}"
echo "   apply_to      = ${APPLY_TO}"
echo "   skip_first    = ${SKIP_FIRST}"
echo "   int_lr        = ${INT_LR}"
echo "   ext_lr        = ${EXT_LR}"
echo "   epochs        = ${EPOCHS}"
echo "   make iters    = ${MAKESET_ITERS}"
echo "   slug          = ${SLUG}"
echo "   CUDA device   = ${CUDA_VISIBLE_DEVICES}"
echo "   checkpoint    = ${CHECKPOINT}"
echo "   run root      = ${RUN_ROOT}"
echo "   PGD enabled   = ${RUN_PGD}"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/5: Training first95_rest80 SIREN backbone"
echo "          -> ${CHECKPOINT}"

python trainer.py \
    --dataset mnist \
    --data-path ../data \
    --device cuda \
    --num-epochs "${EPOCHS}" \
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

python - <<PY
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
print("  variant_args:", ckpt.get("variant_args"))

if ckpt.get("model_name") != expected_model_name:
    print(
        f"ERROR: checkpoint model_name={ckpt.get('model_name')!r} "
        f"does not match expected {expected_model_name!r}",
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo
echo "Step 2/5: Creating functaset"
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
echo "Step 3/5: Training downstream classifier"
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