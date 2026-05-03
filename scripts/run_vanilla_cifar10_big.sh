#!/bin/bash

# Stronger Vanilla CIFAR-10 pipeline:
#   1. meta-train a larger vanilla CIFAR-10 SIREN backbone,
#   2. create a CIFAR-10 functaset with more inner fitting steps,
#   3. train a larger downstream parameter-space classifier,
#   4. evaluate reconstruction quality,
#   5. skip PGD for now.
#
# Motivation:
#   The first CIFAR-10 vanilla run with hidden_dim=256, mod_dim=512 reached
#   only ~18.8 dB PSNR @ 200 fitting steps and ~47% classifier accuracy.
#   This suggests the CIFAR bottleneck is representation capacity.
#
# New setup:
#   hidden_dim = 512
#   mod_dim    = 1024
#   depth      = 10
#   makeset    = 50 inner iterations
#
# Resume / skip training:
#   SKIP_STEP1=1 (default) skips Step 1 if the SIREN checkpoint already exists.
#   SKIP_STEP1=0 runs meta-training from scratch.

set -euo pipefail

SKIP_STEP1="${SKIP_STEP1:-1}"

# ---- architecture ------------------------------------------------------------

HIDDEN_DIM=512
MOD_DIM=1024
DEPTH=10

# ---- training hyperparameters ------------------------------------------------

EPOCHS=8
INT_LR=0.01
EXT_LR=5e-5

# Larger model -> reduce batch size to avoid OOM.
TRAIN_BATCH_SIZE=64

# CIFAR needs more fitting than MNIST.
MAKESET_ITERS=50

CUDA_GPU=1
LOG_SIGMAS_EVERY=50

# Reconstruction eval. 500 is useful to see the ceiling, but costs more.
EVAL_ITERS="20,50,100,200,500"
EVAL_MAX_SAMPLES=2000
EVAL_BATCH_SIZE=32

RUN_PGD=0

# ---- classifier hyperparameters ---------------------------------------------

CLF_LR=0.01
CLF_WIDTH=1024
CLF_DEPTH=4
CLF_DROPOUT=0.3
CLF_BATCH_SIZE=256
CLF_EPOCHS=80

# -----------------------------------------------------------------------------

SLUG="vanilla_cifar10_h${HIDDEN_DIM}_md${MOD_DIM}_d${DEPTH}_e${EPOCHS}_make${MAKESET_ITERS}"

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

echo "== stronger vanilla CIFAR-10 pipeline =="
echo "   dataset       = cifar10"
echo "   variant       = vanilla"
echo "   hidden_dim    = ${HIDDEN_DIM}"
echo "   mod_dim       = ${MOD_DIM}"
echo "   depth         = ${DEPTH}"
echo "   int_lr        = ${INT_LR}"
echo "   ext_lr        = ${EXT_LR}"
echo "   epochs        = ${EPOCHS}"
echo "   train batch   = ${TRAIN_BATCH_SIZE}"
echo "   make iters    = ${MAKESET_ITERS}"
echo "   classifier    = width ${CLF_WIDTH}, depth ${CLF_DEPTH}, dropout ${CLF_DROPOUT}, epochs ${CLF_EPOCHS}"
echo "   slug          = ${SLUG}"
echo "   CUDA device   = ${CUDA_VISIBLE_DEVICES}"
echo "   checkpoint    = ${CHECKPOINT}"
echo "   run root      = ${RUN_ROOT}"
echo "   PGD enabled   = ${RUN_PGD}"
echo "   SKIP_STEP1    = ${SKIP_STEP1}  (1=skip SIREN training if possible)"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
if [[ "${SKIP_STEP1}" -eq 0 ]]; then
    echo "Step 1/5: Training stronger vanilla CIFAR-10 SIREN backbone"
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
        --model-name "${SLUG}" \
        "${VARIANT_FLAGS[@]}" \
        --log-sigmas-every "${LOG_SIGMAS_EVERY}"
else
    echo "Step 1/5: Skipping SIREN backbone training (SKIP_STEP1=${SKIP_STEP1})"
    echo "          expecting checkpoint at ${CHECKPOINT}"
fi

echo
echo "Verifying saved checkpoint"
echo "          -> ${CHECKPOINT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    echo "       Run with SKIP_STEP1=0 to train the backbone, or point MODEL_DIR to an existing run." >&2
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

args = ckpt.get("model_args", {})
expected = {
    "hidden_dim": ${HIDDEN_DIM},
    "mod_dim": ${MOD_DIM},
    "depth": ${DEPTH},
}
for k, v in expected.items():
    if args.get(k) != v:
        print(f"WARNING: model_args[{k!r}]={args.get(k)!r}, expected {v!r}")
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
    --checkpoint "${CHECKPOINT}" \
    --saveroot "${RUN_ROOT}" \
    --device cuda \
    --hidden-dim "${HIDDEN_DIM}" \
    --mod-dim "${MOD_DIM}" \
    --depth "${DEPTH}" \
    --functaset-stem "${SLUG}" \
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
    --hidden-dim "${HIDDEN_DIM}" \
    --mod-dim "${MOD_DIM}" \
    --depth "${DEPTH}" \
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
    echo "ERROR: CIFAR-10 PGD is too expensive for this script. Run PGD separately after reconstruction/classifier validation." >&2
    exit 1
else
    echo "          skipped because RUN_PGD=0"
    PGD_LOG="${RUN_ROOT}/pgd_attack.log"
fi

echo
echo "Done."
echo "SIREN checkpoint : ${CHECKPOINT}"
echo "Functaset        : ${RUN_ROOT}/functaset/${SLUG}_{train,val,test}.pkl"
echo "Classifier       : ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo "Recon eval JSON  : ${RUN_ROOT}/reconstruction_eval.json"
echo "PGD attack log   : skipped"