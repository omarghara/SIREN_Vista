#!/bin/bash

# Spatial Functa CIFAR-10 (paper schedule) + soft-Lipschitz (90% vanilla sigma caps).
#
# Same backbone / functaset schedule as run_spatial_functa_cifar10_subset.sh (PRESET=paper):
#   - SpatialModulatedINR, SIREN, h256, depth 6, freq 10, 50k/10k, e512
#   - Meta-training inner optim: SGD, 3 steps; makeset: SGD, lr 0.01, 3 iters
#
# Variant: soft_lipschitz with hardcoded caps = 0.9 * vanilla_sigmas in
#   variants/soft_lipschitz.py (spatial paper SIREN depth-6 reference).
#
# Downstream classifier: spatial CNN (not MLP), matching the FINER spatial-paper CNN setup.
#
# Pipeline:
#   1. meta-train spatial backbone with soft-Lipschitz penalty
#   2. makeset (SGD phi fit) + combine train+val
#   3. train CNN classifier on flattened phi grid
#
# Override from env examples:
#   CUDA_GPU=1 bash scripts/run_spatial_functa_cifar10_softlip.sh
#   SKIP_TRAINER=1 SKIP_MAKESET=0 bash scripts/run_spatial_functa_cifar10_softlip.sh

set -euo pipefail

PRESET="paper"

# ---- pipeline knobs (env-overridable) ---------------------------------------
SKIP_TRAINER="${SKIP_TRAINER:-0}"
SKIP_MAKESET="${SKIP_MAKESET:-0}"
SKIP_CLASSIFIER="${SKIP_CLASSIFIER:-0}"

# Soft-Lipschitz (L is ignored by hardcoded _collect_layers; kept for CLI compatibility)
SOFT_LIP_CAP="${SOFT_LIP_CAP:-1.0}"
SOFT_LIP_LAMBDA="${SOFT_LIP_LAMBDA:-1e-2}"
SOFT_LIP_APPLY_TO="${SOFT_LIP_APPLY_TO:-sine_and_readout}"
SOFT_LIP_SKIP_FIRST="${SOFT_LIP_SKIP_FIRST:-0}"
SOFT_LIP_CAP_TAG="${SOFT_LIP_CAP_TAG:-cap90}"

# ---- architecture (spatial Functa, paper preset) ----------------------------

LATENT_SPATIAL_DIM=8
LATENT_DIM=16
SPATIAL_INTERP=nearest
MODULATION_TYPE=shift
USE_LOCAL_COORDS=1
COORD_TAG="norm01"

INR_TYPE=siren
HIDDEN_DIM=256
DEPTH=6
FREQ=10.0

EPOCHS=12
INT_LR=0.01
INNER_STEPS=3
META_INNER_OPTIM=sgd
EXT_LR=3e-5
TRAIN_BATCH_SIZE=128

MAKESET_ITERS=3
MAKESET_INNER_OPTIM=sgd
MAKESET_LR=0.01

MOD_DIM=$(( LATENT_SPATIAL_DIM * LATENT_SPATIAL_DIM * LATENT_DIM ))

MAX_TRAIN_SAMPLES=50000
MAX_TEST_SAMPLES=10000

CUDA_GPU="${CUDA_GPU:-0}"
LOG_SIGMAS_EVERY="${LOG_SIGMAS_EVERY:-50}"

# CNN classifier (spatial phi grid)
CLF_TYPE=cnn
CLF_SAVE_SUBDIR=cifar10_cnn_classifier
CNN_WIDTH=128
CLF_DROPOUT=0.1
CLF_LR=0.001
CLF_BATCH_SIZE=256
CLF_EPOCHS=120
CLF_NORMALIZE_PHI=1

# -----------------------------------------------------------------------------

if [[ "${META_INNER_OPTIM}" != "sgd" && "${META_INNER_OPTIM}" != "adam" ]]; then
    echo "ERROR: META_INNER_OPTIM must be 'sgd' or 'adam', got: ${META_INNER_OPTIM}" >&2
    exit 1
fi
if [[ "${MAKESET_INNER_OPTIM}" != "sgd" && "${MAKESET_INNER_OPTIM}" != "adam" ]]; then
    echo "ERROR: MAKESET_INNER_OPTIM must be 'sgd' or 'adam', got: ${MAKESET_INNER_OPTIM}" >&2
    exit 1
fi

EXT_LR_TAG=$(printf "%.0e" "${EXT_LR}")
MAKESET_LR_TAG=$(printf "%.0e" "${MAKESET_LR}")
LAM_TAG=$(printf "%.0e" "${SOFT_LIP_LAMBDA}")
LCOORDS_TAG=$([ "${USE_LOCAL_COORDS}" -eq 1 ] && echo "lc" || echo "gc")

SLUG="functa_like_cifar10_spatial_${PRESET}_${INR_TYPE}_h${HIDDEN_DIM}_md${MOD_DIM}_d${DEPTH}_lat${LATENT_SPATIAL_DIM}x${LATENT_DIM}_freq${FREQ}_${SPATIAL_INTERP}_${LCOORDS_TAG}_${COORD_TAG}_extlr${EXT_LR_TAG}_e${EPOCHS}_inner${INNER_STEPS}_mopt${META_INNER_OPTIM}_adamphi${MAKESET_ITERS}_lr${MAKESET_LR_TAG}_softlip_${SOFT_LIP_CAP_TAG}_lam${LAM_TAG}_${SOFT_LIP_APPLY_TO}_train${MAX_TRAIN_SAMPLES}_test${MAX_TEST_SAMPLES}"

VARIANT_FLAGS=(
    --variant soft_lipschitz
    --soft-lip-cap "${SOFT_LIP_CAP}"
    --soft-lip-lambda "${SOFT_LIP_LAMBDA}"
    --soft-lip-apply-to "${SOFT_LIP_APPLY_TO}"
)
if [[ "${SOFT_LIP_SKIP_FIRST}" -eq 1 ]]; then
    VARIANT_FLAGS+=(--soft-lip-skip-first)
fi

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

INR_FLAGS=( --inr-type "${INR_TYPE}" --freq "${FREQ}" )

MODEL_DIR="model_cifar10/${SLUG}"
CHECKPOINT="${MODEL_DIR}/modSiren.pth"
RUN_ROOT="runs/${SLUG}"
CLF_DIR="${RUN_ROOT}/${CLF_SAVE_SUBDIR}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "============================================================"
echo "Spatial Functa CIFAR-10 + soft-Lipschitz (PRESET=${PRESET})"
echo "dataset            = cifar10"
echo "variant            = soft_lipschitz (${SOFT_LIP_CAP_TAG} hardcoded caps)"
echo "inr_type           = ${INR_TYPE}"
echo "hidden_dim         = ${HIDDEN_DIM}"
echo "mod_dim (flat)     = ${MOD_DIM}"
echo "depth              = ${DEPTH}"
echo "freq (omega0)      = ${FREQ}"
echo "latent grid s,c    = ${LATENT_SPATIAL_DIM} x ${LATENT_DIM}"
echo "meta epochs        = ${EPOCHS}"
echo "meta inner optim   = ${META_INNER_OPTIM}  steps=${INNER_STEPS}  int_lr=${INT_LR}"
echo "make optim         = ${MAKESET_INNER_OPTIM}  iters=${MAKESET_ITERS}  lr=${MAKESET_LR}"
echo "soft_lip lambda    = ${SOFT_LIP_LAMBDA}  apply_to=${SOFT_LIP_APPLY_TO}"
echo "train/test samples = ${MAX_TRAIN_SAMPLES} / ${MAX_TEST_SAMPLES}"
echo "classifier         = ${CLF_TYPE}  save_dir=${CLF_SAVE_SUBDIR}"
echo "slug               = ${SLUG}"
echo "checkpoint         = ${CHECKPOINT}"
echo "run root           = ${RUN_ROOT}"
echo "GPU                = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/3: Training spatial Functa backbone (soft-Lipschitz)"
echo "          -> ${CHECKPOINT}"
echo

if [[ "${SKIP_TRAINER}" == "1" ]]; then
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "ERROR: SKIP_TRAINER=1 but checkpoint missing: ${CHECKPOINT}" >&2
        exit 1
    fi
    echo "[skip] SKIP_TRAINER=1; reusing existing checkpoint."
else
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
        --inner-optim "${META_INNER_OPTIM}" \
        "${INR_FLAGS[@]}" \
        "${SPATIAL_FLAGS[@]}" \
        --model-name "${SLUG}" \
        "${VARIANT_FLAGS[@]}" \
        --log-sigmas-every "${LOG_SIGMAS_EVERY}"
fi

echo
echo "Verifying saved checkpoint"
echo "          -> ${CHECKPOINT}"
echo

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: expected checkpoint was not created: ${CHECKPOINT}" >&2
    exit 1
fi

CKPT_PATH="${CHECKPOINT}" \
EXPECTED_MODEL_NAME="${SLUG}" \
EXPECTED_VARIANT="soft_lipschitz" \
EXPECTED_INR_TYPE="${INR_TYPE}" \
EXPECTED_HIDDEN_DIM="${HIDDEN_DIM}" \
EXPECTED_MOD_DIM="${MOD_DIM}" \
EXPECTED_DEPTH="${DEPTH}" \
EXPECTED_FREQ="${FREQ}" \
EXPECTED_INNER_OPTIM="${META_INNER_OPTIM}" \
EXPECTED_LATENT_SPATIAL_DIM="${LATENT_SPATIAL_DIM}" \
EXPECTED_LATENT_DIM="${LATENT_DIM}" \
REQUESTED_EPOCHS="${EPOCHS}" \
python - <<'PYCKPT'
import os
import sys
import torch

ckpt_path = os.environ["CKPT_PATH"]
expected_model_name = os.environ["EXPECTED_MODEL_NAME"]
expected_variant = os.environ["EXPECTED_VARIANT"]
ckpt = torch.load(ckpt_path, map_location="cpu")

print("checkpoint metadata:")
print("  epoch:", ckpt.get("epoch"))
print("  loss:", ckpt.get("loss"))
print("  variant:", ckpt.get("variant"))
print("  model_name:", ckpt.get("model_name"))

if ckpt.get("model_name") != expected_model_name:
    print(
        f"ERROR: checkpoint model_name={ckpt.get('model_name')!r} "
        f"does not match expected {expected_model_name!r}",
        file=sys.stderr,
    )
    sys.exit(1)

if ckpt.get("variant") != expected_variant:
    print(
        f"ERROR: checkpoint variant={ckpt.get('variant')!r} "
        f"expected {expected_variant!r}",
        file=sys.stderr,
    )
    sys.exit(1)

model_args = ckpt.get("model_args", {})
expected_lat_s = int(os.environ["EXPECTED_LATENT_SPATIAL_DIM"])
expected_lat_c = int(os.environ["EXPECTED_LATENT_DIM"])
expected_mod_dim = int(os.environ["EXPECTED_MOD_DIM"])

checks = {
    "dataset": "cifar10",
    "inr_type": os.environ["EXPECTED_INR_TYPE"],
    "hidden_dim": int(os.environ["EXPECTED_HIDDEN_DIM"]),
    "mod_dim": expected_mod_dim,
    "depth": int(os.environ["EXPECTED_DEPTH"]),
    "freq": float(os.environ["EXPECTED_FREQ"]),
    "inner_optim": os.environ["EXPECTED_INNER_OPTIM"],
    "spatial_modulation": True,
    "latent_spatial_dim": expected_lat_s,
    "latent_dim": expected_lat_c,
    "is_spatial": True,
    "phi_numel": expected_mod_dim,
}

for key, expected in checks.items():
    got = model_args.get(key)
    if got != expected:
        print(f"ERROR: model_args[{key!r}]={got!r}, expected {expected!r}", file=sys.stderr)
        sys.exit(1)

phi_shape = tuple(model_args.get("phi_shape", ()))
expected_shape = (expected_lat_s, expected_lat_s, expected_lat_c)
if phi_shape != expected_shape:
    print(f"ERROR: phi_shape={phi_shape}, expected {expected_shape}", file=sys.stderr)
    sys.exit(1)

print("Checkpoint verified.")

summary_path = os.path.join(os.path.dirname(ckpt_path), "checkpoint_summary.md")
variant_args = ckpt.get("variant_args", {}) or {}
lines = [
    "# Backbone checkpoint summary",
    "",
    f"- **path**: `{ckpt_path}`",
    f"- **model_name**: `{ckpt.get('model_name')}`",
    f"- **variant**: `{ckpt.get('variant')}`",
    f"- **epoch (best)**: {ckpt.get('epoch')}",
    f"- **loss (best mean outer loss)**: {ckpt.get('loss')}",
    f"- **num_epochs requested**: {os.environ['REQUESTED_EPOCHS']}",
    "",
    "## model_args",
    "",
]
for k in sorted(model_args.keys()):
    lines.append(f"- `{k}`: {model_args[k]!r}")
if variant_args:
    lines += ["", "## variant_args", ""]
    for k in sorted(variant_args.keys()):
        lines.append(f"- `{k}`: {variant_args[k]!r}")
lines.append("")
with open(summary_path, "w") as fh:
    fh.write("\n".join(lines))
print(f"Wrote backbone summary: {summary_path}")
PYCKPT

echo
echo "Step 2/3: Creating ${MAX_TRAIN_SAMPLES}-train / ${MAX_TEST_SAMPLES}-test spatial functaset (SGD phi, ${MAKESET_ITERS} iters)"
echo "          -> ${RUN_ROOT}/functaset/${SLUG}_{train,val,test}.pkl"
echo

if [[ "${SKIP_MAKESET}" == "1" ]]; then
    if [[ ! -d "${RUN_ROOT}/functaset" ]]; then
        echo "ERROR: SKIP_MAKESET=1 but ${RUN_ROOT}/functaset is missing." >&2
        exit 1
    fi
    echo "[skip] SKIP_MAKESET=1; reusing existing functaset directory."
else
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
fi

echo
echo "Combining train + val into one ${MAX_TRAIN_SAMPLES}-sample train set"
echo

FUNCTASET_ROOT="${RUN_ROOT}/functaset" \
SLUG="${SLUG}" \
EXPECTED_LATENT_SPATIAL_DIM="${LATENT_SPATIAL_DIM}" \
EXPECTED_LATENT_DIM="${LATENT_DIM}" \
EXPECTED_MOD_DIM="${MOD_DIM}" \
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES}" \
python - <<'PYCOMBINE'
import os
import sys
import joblib
from collections import Counter

root = os.environ["FUNCTASET_ROOT"]
slug = os.environ["SLUG"]
max_train = os.environ["MAX_TRAIN_SAMPLES"]
expected_lat_s = int(os.environ["EXPECTED_LATENT_SPATIAL_DIM"])
expected_lat_c = int(os.environ["EXPECTED_LATENT_DIM"])
expected_shape = (expected_lat_s, expected_lat_s, expected_lat_c)
expected_numel = int(os.environ["EXPECTED_MOD_DIM"])

train = joblib.load(f"{root}/{slug}_train.pkl")
val = joblib.load(f"{root}/{slug}_val.pkl")
test = joblib.load(f"{root}/{slug}_test.pkl")

combined = train + val

out_path = f"{root}/{slug}_train_all{max_train}.pkl"
joblib.dump(combined, out_path)

print("train:", len(train), Counter([x["label"] for x in train]))
print("val:", len(val), Counter([x["label"] for x in val]))
print("combined:", len(combined), Counter([x["label"] for x in combined]))
print("test:", len(test), Counter([x["label"] for x in test]))
first = combined[0]
print("first modul shape:", first["modul"].shape, "is_spatial:", first.get("is_spatial"))
print("saved:", out_path)

actual_shape = tuple(first["modul"].shape)
if actual_shape != expected_shape:
    print(
        f"ERROR: combined modul shape={actual_shape}, expected {expected_shape}.",
        file=sys.stderr,
    )
    sys.exit(1)

flat = first["modul"].reshape(-1).shape[0]
if flat != expected_numel:
    print(f"ERROR: flattened modul size={flat}, expected {expected_numel}.", file=sys.stderr)
    sys.exit(1)
PYCOMBINE

echo
echo "Step 3/3: Training downstream spatial CNN classifier"
echo "          -> ${CLF_DIR}/best_classifier.pth"
echo

if [[ "${SKIP_CLASSIFIER}" == "1" ]]; then
    echo "[skip] SKIP_CLASSIFIER=1"
else
    CLF_FLAGS=(
        --dataset cifar10
        --classifier-type "${CLF_TYPE}"
        --functaset-path-train "${RUN_ROOT}/functaset/${SLUG}_train_all${MAX_TRAIN_SAMPLES}.pkl"
        --functaset-path-test "${RUN_ROOT}/functaset/${SLUG}_test.pkl"
        --latent-spatial-dim "${LATENT_SPATIAL_DIM}"
        --latent-dim "${LATENT_DIM}"
        --mod-dim "${MOD_DIM}"
        --cnn-width "${CNN_WIDTH}"
        --dropout "${CLF_DROPOUT}"
        --lr "${CLF_LR}"
        --batch-size "${CLF_BATCH_SIZE}"
        --num-epochs "${CLF_EPOCHS}"
        --device cuda
        --save-dir "${CLF_DIR}"
    )
    if [[ "${CLF_NORMALIZE_PHI}" -eq 1 ]]; then
        CLF_FLAGS+=(--normalize-phi)
    fi

    python train_classifier.py "${CLF_FLAGS[@]}"
fi

echo
echo "Writing classifier and run summaries"
echo

RUN_ROOT="${RUN_ROOT}" \
CKPT_PATH="${CHECKPOINT}" \
SLUG="${SLUG}" \
CLF_DIR="${CLF_DIR}" \
CLF_EPOCHS="${CLF_EPOCHS}" \
CLF_TYPE="${CLF_TYPE}" \
CNN_WIDTH="${CNN_WIDTH}" \
CLF_DROPOUT="${CLF_DROPOUT}" \
CLF_LR="${CLF_LR}" \
CLF_BATCH_SIZE="${CLF_BATCH_SIZE}" \
MOD_DIM="${MOD_DIM}" \
python - <<'PYSUMMARY'
import os
import sys
import numpy as np
import torch

run_root = os.environ["RUN_ROOT"]
ckpt_path = os.environ["CKPT_PATH"]
slug = os.environ["SLUG"]
clf_dir = os.environ["CLF_DIR"]
dataset = "cifar10"
clf_path = os.path.join(clf_dir, "best_classifier.pth")
acc_npy = os.path.join(clf_dir, "classifier_acc.npy")

if not os.path.isfile(clf_path):
    print(f"WARNING: classifier checkpoint missing: {clf_path}", file=sys.stderr)
    sys.exit(0)

clf = torch.load(clf_path, map_location="cpu")
best_acc = float(clf.get("accuracy", float("nan")))
best_epoch = clf.get("epoch")

top1_series = top5_series = None
if os.path.isfile(acc_npy):
    arr = np.load(acc_npy)
    if arr.ndim == 2 and arr.shape[0] >= 2:
        top1_series = arr[0].tolist()
        top5_series = arr[1].tolist()

lines = [
    "# Classifier checkpoint summary",
    "",
    f"- **path**: `{clf_path}`",
    f"- **dataset**: {dataset}",
    f"- **backbone slug**: `{slug}`",
    f"- **classifier type**: {os.environ['CLF_TYPE']}",
    f"- **best top-1 accuracy**: {best_acc:.2f}%",
    f"- **best epoch**: {best_epoch}",
    f"- **num epochs trained**: {os.environ['CLF_EPOCHS']}",
    f"- **cnn width**: {os.environ['CNN_WIDTH']}",
    f"- **dropout**: {os.environ['CLF_DROPOUT']}",
    f"- **lr**: {os.environ['CLF_LR']}",
    f"- **batch size**: {os.environ['CLF_BATCH_SIZE']}",
    f"- **mod_dim (flat)**: {os.environ['MOD_DIM']}",
]
if top1_series is not None:
    lines += [
        "",
        f"- **final top-1**: {top1_series[-1]:.2f}%",
        f"- **final top-5**: {top5_series[-1]:.2f}%",
        f"- **max top-5**: {max(top5_series):.2f}%",
    ]
lines.append("")
with open(os.path.join(clf_dir, "checkpoint_summary.md"), "w") as fh:
    fh.write("\n".join(lines))
print(f"Wrote classifier summary: {os.path.join(clf_dir, 'checkpoint_summary.md')}")

ckpt = torch.load(ckpt_path, map_location="cpu") if os.path.isfile(ckpt_path) else {}
run_summary = [
    "# Run summary",
    "",
    f"- **slug**: `{slug}`",
    f"- **run root**: `{run_root}`",
    f"- **variant**: `{ckpt.get('variant')}`",
    "",
    "## Backbone",
    "",
    f"- **checkpoint**: `{ckpt_path}`",
    f"- **best epoch**: {ckpt.get('epoch')}",
    f"- **best loss (mean outer)**: {ckpt.get('loss')}",
    "",
    "## Classifier",
    "",
    f"- **type**: {os.environ['CLF_TYPE']}",
    f"- **checkpoint**: `{clf_path}`",
    f"- **best top-1 accuracy**: {best_acc:.2f}%",
    f"- **best epoch**: {best_epoch}",
    "",
]
out = os.path.join(run_root, "run_summary.md")
with open(out, "w") as fh:
    fh.write("\n".join(run_summary))
print(f"Wrote run summary: {out}")
PYSUMMARY

echo
echo "============================================================"
echo "Done."
echo "Spatial checkpoint : ${CHECKPOINT}"
echo "Backbone summary   : $(dirname "${CHECKPOINT}")/checkpoint_summary.md"
echo "Functaset train    : ${RUN_ROOT}/functaset/${SLUG}_train_all${MAX_TRAIN_SAMPLES}.pkl"
echo "Functaset test     : ${RUN_ROOT}/functaset/${SLUG}_test.pkl"
echo "CNN classifier     : ${CLF_DIR}/best_classifier.pth"
echo "Classifier summary : ${CLF_DIR}/checkpoint_summary.md"
echo "Run summary        : ${RUN_ROOT}/run_summary.md"
echo "============================================================"
