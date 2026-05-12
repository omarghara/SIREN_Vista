#!/bin/bash

# Spatial Functa CIFAR-10 pipeline (paper-aligned first reproduction):
#   1. meta-train a SpatialModulatedINR (SIREN backbone, 1-NN, shift-only),
#   2. create a 5000-train / 1000-test spatial functaset (phi has shape (s,s,c)),
#   3. combine train+val back into one training set,
#   4. train downstream parameter-space MLP classifier on flattened (s*s*c)
#      modulations,
#   5. run reconstruction eval (slow per-image inner loop; v1 is correct first,
#      batching can come later).
#
# Reference: From Data to Functa (Dupont et al. 2022, arXiv:2201.12204),
# Sec. 4 + Appendix C.1, Table 4 (1-NN row, CIFAR-10).
#
# This script intentionally does not change any non-spatial training logic
# in this repo: all spatial behavior is gated behind --spatial-modulation.

set -euo pipefail

# ---- architecture (paper Table 4, 1-NN row) ---------------------------------
HIDDEN_DIM=256
DEPTH=6
INR_TYPE=siren            # 'siren' | 'fourier_siren' | 'finer' | 'fourier_lsa'
SIREN_FREQ=30.0           # ω0 of every hidden sine layer

# Spatial latent grid: phi has shape (LATENT_SPATIAL_DIM, LATENT_SPATIAL_DIM, LATENT_DIM)
LATENT_SPATIAL_DIM=8      # s ; each cell covers CIFAR's 32/8 = 4 pixels per side
LATENT_DIM=16             # c
SPATIAL_INTERP=nearest
MODULATION_TYPE=shift
USE_LOCAL_COORDS=1        # set 0 to feed global coords instead of per-cell local coords

# Flat phi numel = s*s*c (used as MLP classifier input). 8*8*16 = 1024
MOD_DIM=$(( LATENT_SPATIAL_DIM * LATENT_SPATIAL_DIM * LATENT_DIM ))

# ---- meta-training hyperparameters ------------------------------------------
EPOCHS=10
INT_LR=0.01
INNER_STEPS=3
INNER_OPTIM=sgd
EXT_LR=3e-5
TRAIN_BATCH_SIZE=128

CUDA_GPU=0
LOG_SIGMAS_EVERY=100

# ---- functaset creation -----------------------------------------------------
MAKESET_ITERS=200
MAKESET_INNER_OPTIM=adam
MAKESET_LR=0.003

MAX_TRAIN_SAMPLES=5000
MAX_TEST_SAMPLES=1000

# ---- classifier -------------------------------------------------------------
CLF_LR=0.001
CLF_WIDTH=512
CLF_DEPTH=2
CLF_DROPOUT=0.5
CLF_BATCH_SIZE=256
CLF_EPOCHS=120

# ---- reconstruction eval ----------------------------------------------------
EVAL_ITER_CHECKPOINTS="5,20,50,100,200"
EVAL_INNER_LR=0.01
EVAL_BATCH_SIZE=32
EVAL_MAX_SAMPLES=1000

# -----------------------------------------------------------------------------
# Follow-up experiment (commented): deeper backbone (depth=10) once the
# depth=6 reproduction is validated end-to-end. Uncomment to enable.
# HIDDEN_DIM=256
# DEPTH=10
# -----------------------------------------------------------------------------

EXT_LR_TAG=$(printf "%.0e" "${EXT_LR}")
LCOORDS_TAG=$([ "${USE_LOCAL_COORDS}" -eq 1 ] && echo "lc" || echo "gc")

SLUG="spatial_cifar10_${INR_TYPE}_h${HIDDEN_DIM}_d${DEPTH}_lat${LATENT_SPATIAL_DIM}x${LATENT_SPATIAL_DIM}x${LATENT_DIM}_${SPATIAL_INTERP}_${LCOORDS_TAG}_w0_${SIREN_FREQ}_extlr${EXT_LR_TAG}_e${EPOCHS}_inner${INNER_STEPS}_${INNER_OPTIM}"

VARIANT_FLAGS=(
    --variant vanilla
)

SPATIAL_FLAGS=(
    --spatial-modulation
    --latent-spatial-dim "${LATENT_SPATIAL_DIM}"
    --latent-dim         "${LATENT_DIM}"
    --spatial-interp     "${SPATIAL_INTERP}"
    --modulation-type    "${MODULATION_TYPE}"
)
if [[ "${USE_LOCAL_COORDS}" -eq 1 ]]; then
    SPATIAL_FLAGS+=(--use-local-coords)
fi

INR_FLAGS=(
    --inr-type   "${INR_TYPE}"
    --siren-freq "${SIREN_FREQ}"
)

MODEL_DIR="model_cifar10/${SLUG}"
CHECKPOINT="${MODEL_DIR}/modSiren.pth"
RUN_ROOT="runs/${SLUG}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "============================================================"
echo "Spatial Functa CIFAR-10 pipeline"
echo "dataset            = cifar10"
echo "inr_type           = ${INR_TYPE}"
echo "hidden_dim         = ${HIDDEN_DIM}"
echo "depth              = ${DEPTH}"
echo "siren_freq (w0)    = ${SIREN_FREQ}"
echo "latent grid (s,c)  = ${LATENT_SPATIAL_DIM} x ${LATENT_SPATIAL_DIM} x ${LATENT_DIM}"
echo "phi_numel (mod_dim)= ${MOD_DIM}"
echo "interp             = ${SPATIAL_INTERP}"
echo "use_local_coords   = ${USE_LOCAL_COORDS}"
echo "modulation_type    = ${MODULATION_TYPE}"
echo "meta epochs        = ${EPOCHS}"
echo "meta int_lr        = ${INT_LR}"
echo "meta ext_lr        = ${EXT_LR}"
echo "inner steps        = ${INNER_STEPS}"
echo "inner optim        = ${INNER_OPTIM}"
echo "make iters         = ${MAKESET_ITERS}"
echo "make optim         = ${MAKESET_INNER_OPTIM}"
echo "make lr            = ${MAKESET_LR}"
echo "train samples      = ${MAX_TRAIN_SAMPLES}"
echo "test samples       = ${MAX_TEST_SAMPLES}"
echo "slug               = ${SLUG}"
echo "checkpoint         = ${CHECKPOINT}"
echo "run root           = ${RUN_ROOT}"
echo "GPU                = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/5: Training Spatial Functa CIFAR-10 backbone"
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
    --inner-optim "${INNER_OPTIM}" \
    "${INR_FLAGS[@]}" \
    "${SPATIAL_FLAGS[@]}" \
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
import sys, torch

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
    print(f"ERROR: checkpoint model_name={ckpt.get('model_name')!r} "
          f"does not match expected {expected_model_name!r}",
          file=sys.stderr)
    sys.exit(1)

model_args = ckpt.get("model_args", {})

checks = {
    "dataset": "cifar10",
    "inr_type": "${INR_TYPE}",
    "hidden_dim": ${HIDDEN_DIM},
    "depth": ${DEPTH},
    "spatial_modulation": True,
    "latent_spatial_dim": ${LATENT_SPATIAL_DIM},
    "latent_dim": ${LATENT_DIM},
    "is_spatial": True,
    "phi_numel": ${MOD_DIM},
}

for key, expected in checks.items():
    got = model_args.get(key)
    if got != expected:
        print(f"ERROR: model_args[{key!r}]={got!r}, expected {expected!r}", file=sys.stderr)
        sys.exit(1)

phi_shape = tuple(model_args.get("phi_shape", ()))
expected_shape = (${LATENT_SPATIAL_DIM}, ${LATENT_SPATIAL_DIM}, ${LATENT_DIM})
if phi_shape != expected_shape:
    print(f"ERROR: phi_shape={phi_shape}, expected {expected_shape}", file=sys.stderr)
    sys.exit(1)

print("Checkpoint verified.")
PYCKPT

echo
echo "Step 2/5: Creating ${MAX_TRAIN_SAMPLES}-train / ${MAX_TEST_SAMPLES}-test CIFAR-10 spatial functaset"
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
    "${SPATIAL_FLAGS[@]}" \
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
print("first modul shape:", combined[0]["modul"].shape, "is_spatial:", combined[0].get("is_spatial"))
print("saved:", out_path)
PYCOMBINE

echo
echo "Step 3/5: Training downstream CIFAR-10 MLP classifier on flattened spatial modulations"
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
    --classifier-type mlp \
    --device cuda

popd > /dev/null

echo
echo "Step 4/5: Reconstruction eval (slow per-image inner loop for spatial models; v1)"
echo "          -> ${RUN_ROOT}/reconstruction_eval.json"
echo

python evaluate_reconstruction.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset cifar10 \
    --data-path ../data \
    --device cuda \
    --split test \
    --iter-checkpoints "${EVAL_ITER_CHECKPOINTS}" \
    --inner-lr "${EVAL_INNER_LR}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --max-samples "${EVAL_MAX_SAMPLES}" \
    --output "${RUN_ROOT}/reconstruction_eval.json" \
    "${INR_FLAGS[@]}" \
    "${SPATIAL_FLAGS[@]}" \
    "${VARIANT_FLAGS[@]}"

echo
echo "Step 5/5: Done."
echo "============================================================"
echo "Spatial checkpoint    : ${CHECKPOINT}"
echo "Functaset train (all) : ${RUN_ROOT}/functaset/${SLUG}_train_all${MAX_TRAIN_SAMPLES}.pkl"
echo "Functaset test        : ${RUN_ROOT}/functaset/${SLUG}_test.pkl"
echo "Classifier            : ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo "Reconstruction eval   : ${RUN_ROOT}/reconstruction_eval.json"
echo "============================================================"
