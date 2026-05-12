#!/bin/bash

# Spatial Functa CIFAR-10 subset pipeline (same layout as run_fourier_cifar10.sh / FINER script):
#   1. meta-train SpatialModulatedINR (global coords + 1-NN latent grid + per-pixel shifts),
#   2. create 5000-train / 1000-test spatial functaset (modul shape (s,s,c) per image),
#   3. combine train+val into one training set,
#   4. train downstream MLP on flattened modulations (B, s*s*c).
#
# No reconstruction eval.
# No PGD attack.
#
# Reference: Dupont et al. 2022 (From Data to Functa), Table 4 (1-NN row uses s=8, c=16).
# You can scale HIDDEN_DIM / DEPTH like the FINER script; MOD_DIM is always s*s*c.

set -euo pipefail

# ---- architecture ------------------------------------------------------------

HIDDEN_DIM=512
DEPTH=10

INR_TYPE=finer            # siren | fourier_siren | finer | fourier_lsa
COORD_TAG="norm01"

SIREN_FREQ=30.0

# Spatial latent grid: phi shape (LATENT_SPATIAL_DIM, LATENT_SPATIAL_DIM, LATENT_DIM)
LATENT_SPATIAL_DIM=8
LATENT_DIM=16
SPATIAL_INTERP=nearest
MODULATION_TYPE=shift
USE_LOCAL_COORDS=1

MOD_DIM=$(( LATENT_SPATIAL_DIM * LATENT_SPATIAL_DIM * LATENT_DIM ))

# Fourier path (only used when INR_TYPE=fourier_siren or fourier_lsa)
FOURIER_NUM_FREQS=64
FOURIER_SIGMA=10.0
FOURIER_INCLUDE_INPUT=0

# FINER path (only used when INR_TYPE=finer)
FINER_FREQ=30.0
FINER_FIRST_BIAS_SCALE=2.0
FINER_SCALE_REQ_GRAD=0

# LSA path (only used when INR_TYPE=fourier_lsa)
LSA_NUM_FREQS=64
LSA_SIGMA=10.0

# ---- meta-training hyperparameters ------------------------------------------

EPOCHS=5
INT_LR=0.01
INNER_STEPS=3

EXT_LR=1e-5
TRAIN_BATCH_SIZE=32

CUDA_GPU=0
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
MAKESET_LR_TAG="3e-03"
LCOORDS_TAG=$([ "${USE_LOCAL_COORDS}" -eq 1 ] && echo "lc" || echo "gc")

SLUG="functa_like_cifar10_spatial_${INR_TYPE}_h${HIDDEN_DIM}_md${MOD_DIM}_d${DEPTH}_lat${LATENT_SPATIAL_DIM}x${LATENT_DIM}_${SPATIAL_INTERP}_${LCOORDS_TAG}_${COORD_TAG}_extlr${EXT_LR_TAG}_e${EPOCHS}_inner${INNER_STEPS}_adamphi${MAKESET_ITERS}_lr${MAKESET_LR_TAG}_train${MAX_TRAIN_SAMPLES}_test${MAX_TEST_SAMPLES}"

VARIANT_FLAGS=(
    --variant vanilla
)

SPATIAL_FLAGS=(
    --spatial-modulation
    --latent-spatial-dim "${LATENT_SPATIAL_DIM}"
    --latent-dim "${LATENT_DIM}"
    --spatial-interp "${SPATIAL_INTERP}"
    --modulation-type "${MODULATION_TYPE}"
)
if [[ "${USE_LOCAL_COORDS}" -eq 1 ]]; then
    SPATIAL_FLAGS+=(--use-local-coords)
fi

INR_FLAGS=( --inr-type "${INR_TYPE}" )

case "${INR_TYPE}" in
    siren)
        INR_FLAGS+=( --siren-freq "${SIREN_FREQ}" )
        ;;
    fourier_siren)
        INR_FLAGS+=( --siren-freq "${SIREN_FREQ}" )
        INR_FLAGS+=( --fourier-num-freqs "${FOURIER_NUM_FREQS}" --fourier-sigma "${FOURIER_SIGMA}" )
        if [[ "${FOURIER_INCLUDE_INPUT}" -eq 1 ]]; then
            INR_FLAGS+=( --fourier-include-input )
        fi
        ;;
    finer)
        INR_FLAGS+=( --finer-freq "${FINER_FREQ}" --finer-first-bias-scale "${FINER_FIRST_BIAS_SCALE}" )
        if [[ "${FINER_SCALE_REQ_GRAD}" -eq 1 ]]; then
            INR_FLAGS+=( --finer-scale-req-grad )
        fi
        ;;
    fourier_lsa)
        INR_FLAGS+=( --lsa-num-freqs "${LSA_NUM_FREQS}" --lsa-sigma "${LSA_SIGMA}" )
        INR_FLAGS+=( --fourier-num-freqs "${FOURIER_NUM_FREQS}" --fourier-sigma "${FOURIER_SIGMA}" )
        if [[ "${FOURIER_INCLUDE_INPUT}" -eq 1 ]]; then
            INR_FLAGS+=( --fourier-include-input )
        fi
        ;;
    *)
        echo "ERROR: unsupported INR_TYPE=${INR_TYPE}" >&2
        exit 1
        ;;
esac

MODEL_DIR="model_cifar10/${SLUG}"
CHECKPOINT="${MODEL_DIR}/modSiren.pth"
RUN_ROOT="runs/${SLUG}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "============================================================"
echo "Spatial Functa CIFAR-10 subset pipeline"
echo "dataset            = cifar10"
echo "inr_type           = ${INR_TYPE}"
echo "hidden_dim         = ${HIDDEN_DIM}"
echo "mod_dim (flat)     = ${MOD_DIM}"
echo "depth              = ${DEPTH}"
echo "latent grid s,c    = ${LATENT_SPATIAL_DIM} x ${LATENT_DIM}"
echo "spatial_interp     = ${SPATIAL_INTERP}"
echo "use_local_coords   = ${USE_LOCAL_COORDS}"
echo "meta epochs        = ${EPOCHS}"
echo "meta int_lr        = ${INT_LR}"
echo "meta ext_lr        = ${EXT_LR}"
echo "inner steps        = ${INNER_STEPS}"
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
echo "Step 1/3: Training Spatial Functa CIFAR-10 backbone"
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
    "inr_type": "${INR_TYPE}",
    "hidden_dim": ${HIDDEN_DIM},
    "mod_dim": ${MOD_DIM},
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
echo "Step 2/3: Creating ${MAX_TRAIN_SAMPLES}-train / ${MAX_TEST_SAMPLES}-test CIFAR-10 spatial functaset"
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
echo "Step 3/3: Training downstream CIFAR-10 classifier (flattened spatial phi)"
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
echo "============================================================"
echo "Done."
echo "Spatial checkpoint : ${CHECKPOINT}"
echo "Functaset train    : ${RUN_ROOT}/functaset/${SLUG}_train_all${MAX_TRAIN_SAMPLES}.pkl"
echo "Functaset test     : ${RUN_ROOT}/functaset/${SLUG}_test.pkl"
echo "Classifier         : ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo "============================================================"
