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
#
# ---- PRESET SELECTOR --------------------------------------------------------
# Set PRESET to choose a configuration block:
#
#   PRESET=current   (default) — FINER backbone, hidden=512, depth=10, freq=60, ext_lr=1e-5
#                                Large model, 5 epochs.  Good for quick ablation.
#
#   PRESET=paper     — Spatial Functa with FINER backbone (same schedule as Table 4 row):
#                       hidden=256, depth=6, freq=10, finer_first_bias_scale=2,
#                       ext_lr=3e-5, batch=128
#                       ~511 epochs ≈ 200k outer updates on CIFAR-10 (50k images / 128 per batch)
#                       makeset inner steps = 3 (same as trainer inner steps)
#
# Override from env:  PRESET=paper bash scripts/run_spatial_functa_cifar10_subset.sh
# ---------------------------------------------------------------------------

PRESET="paper"

#RESET="${PRESET:-current}"

set -euo pipefail

# ---- pipeline knobs (env-overridable) ---------------------------------------
# SKIP_TRAINER=1 reuses an existing modSiren.pth at ${CHECKPOINT}.
# SKIP_MAKESET=1 reuses an existing functaset directory at ${RUN_ROOT}/functaset.
SKIP_TRAINER="${SKIP_TRAINER:-0}"
SKIP_MAKESET="${SKIP_MAKESET:-0}"

# ---- architecture (shared defaults, overridden per-preset below) ------------

# Spatial latent grid: phi shape (LATENT_SPATIAL_DIM, LATENT_SPATIAL_DIM, LATENT_DIM)
LATENT_SPATIAL_DIM=8
LATENT_DIM=16
SPATIAL_INTERP=nearest
MODULATION_TYPE=shift
USE_LOCAL_COORDS=1

COORD_TAG="norm01"

# Fourier path (only used when INR_TYPE=fourier_siren or fourier_lsa)
FOURIER_NUM_FREQS=64
FOURIER_SIGMA=10.0
FOURIER_INCLUDE_INPUT=0

# FINER path (only used when INR_TYPE=finer)
FINER_FIRST_BIAS_SCALE=2.0
FINER_SCALE_REQ_GRAD=0

# LSA path (only used when INR_TYPE=fourier_lsa)
LSA_NUM_FREQS=64
LSA_SIGMA=10.0

# ---- per-preset configuration -----------------------------------------------

if [[ "${PRESET}" == "paper" ]]; then
    # Paper-like schedule (Table 4 hyperparams) with FINER instead of SIREN;
    # FINER_FIRST_BIAS_SCALE=2.0 from shared defaults above.
    # ~511 epochs ≈ 200k outer updates (50000 images / batch 128 * 511 ≈ 199 900)
    INR_TYPE=siren
    HIDDEN_DIM=256
    DEPTH=6
    FREQ=10.0
   #FINER_FIRST_BIAS_SCALE=2.0

    EPOCHS=512               # ≈ 200k outer updates at batch_size=128 on CIFAR-10
    INT_LR=0.01              # inner lr (phi optimisation in meta-training)
    INNER_STEPS=3
    META_INNER_OPTIM=sgd
    EXT_LR=3e-5              # outer Adam lr on backbone
    TRAIN_BATCH_SIZE=128

    MAKESET_ITERS=3          # inner steps when building functaset
    MAKESET_INNER_OPTIM=sgd
    MAKESET_LR=0.01

    CLF_LR=0.001
    CLF_WIDTH=512
    CLF_DEPTH=2
    CLF_DROPOUT=0.5
    CLF_BATCH_SIZE=256
    CLF_EPOCHS=120

elif [[ "${PRESET}" == "current" ]]; then
    # Larger FINER model — quick local baseline
    INR_TYPE=finer
    HIDDEN_DIM=512
    DEPTH=10
    FREQ=60.0

    EPOCHS=5
    INT_LR=0.01
    INNER_STEPS=3
    META_INNER_OPTIM=sgd
    EXT_LR=1e-5
    TRAIN_BATCH_SIZE=32

    MAKESET_ITERS=200
    MAKESET_INNER_OPTIM=adam
    MAKESET_LR=0.003

    CLF_LR=0.001
    CLF_WIDTH=512
    CLF_DEPTH=2
    CLF_DROPOUT=0.5
    CLF_BATCH_SIZE=256
    CLF_EPOCHS=120

else
    echo "ERROR: unknown PRESET='${PRESET}'. Use 'current' or 'paper'." >&2
    exit 1
fi

MOD_DIM=$(( LATENT_SPATIAL_DIM * LATENT_SPATIAL_DIM * LATENT_DIM ))

MAX_TRAIN_SAMPLES=50000
MAX_TEST_SAMPLES=10000

CUDA_GPU=0
LOG_SIGMAS_EVERY=50

# -----------------------------------------------------------------------------

if [[ "${META_INNER_OPTIM}" != "sgd" && "${META_INNER_OPTIM}" != "adam" ]]; then
    echo "ERROR: META_INNER_OPTIM must be 'sgd' or 'adam', got: ${META_INNER_OPTIM}" >&2
    exit 1
fi

EXT_LR_TAG=$(printf "%.0e" "${EXT_LR}")
MAKESET_LR_TAG=$(printf "%.0e" "${MAKESET_LR}")
LCOORDS_TAG=$([ "${USE_LOCAL_COORDS}" -eq 1 ] && echo "lc" || echo "gc")

SLUG="functa_like_cifar10_spatial_${PRESET}_${INR_TYPE}_h${HIDDEN_DIM}_md${MOD_DIM}_d${DEPTH}_lat${LATENT_SPATIAL_DIM}x${LATENT_DIM}_freq${FREQ}_${SPATIAL_INTERP}_${LCOORDS_TAG}_${COORD_TAG}_extlr${EXT_LR_TAG}_e${EPOCHS}_inner${INNER_STEPS}_mopt${META_INNER_OPTIM}_adamphi${MAKESET_ITERS}_lr${MAKESET_LR_TAG}_train${MAX_TRAIN_SAMPLES}_test${MAX_TEST_SAMPLES}"

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
        INR_FLAGS+=( --freq "${FREQ}" )
        ;;
    fourier_siren)
        INR_FLAGS+=( --freq "${FREQ}" )
        INR_FLAGS+=( --fourier-num-freqs "${FOURIER_NUM_FREQS}" --fourier-sigma "${FOURIER_SIGMA}" )
        if [[ "${FOURIER_INCLUDE_INPUT}" -eq 1 ]]; then
            INR_FLAGS+=( --fourier-include-input )
        fi
        ;;
    finer)
        INR_FLAGS+=( --freq "${FREQ}" --finer-first-bias-scale "${FINER_FIRST_BIAS_SCALE}" )
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
echo "Spatial Functa CIFAR-10 subset pipeline   (PRESET=${PRESET})"
echo "dataset            = cifar10"
echo "inr_type           = ${INR_TYPE}"
echo "hidden_dim         = ${HIDDEN_DIM}"
echo "mod_dim (flat)     = ${MOD_DIM}"
echo "depth              = ${DEPTH}"
echo "freq (omega0)      = ${FREQ}"
echo "latent grid s,c    = ${LATENT_SPATIAL_DIM} x ${LATENT_DIM}"
echo "spatial_interp     = ${SPATIAL_INTERP}"
echo "use_local_coords   = ${USE_LOCAL_COORDS}"
echo "meta epochs        = ${EPOCHS}"
echo "meta int_lr        = ${INT_LR}"
echo "meta ext_lr        = ${EXT_LR}"
echo "meta inner optim   = ${META_INNER_OPTIM}  (trainer.py --inner-optim)"
echo "inner steps        = ${INNER_STEPS}"
echo "train batch size   = ${TRAIN_BATCH_SIZE}"
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
echo "Step 2/3: Creating ${MAX_TRAIN_SAMPLES}-train / ${MAX_TEST_SAMPLES}-test CIFAR-10 spatial functaset"
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
        f"ERROR: combined modul shape={actual_shape}, expected {expected_shape}. "
        "Did makeset.py load a stale checkpoint with a different latent grid?",
        file=sys.stderr,
    )
    sys.exit(1)

flat = first["modul"].reshape(-1).shape[0]
if flat != expected_numel:
    print(f"ERROR: flattened modul size={flat}, expected {expected_numel}.", file=sys.stderr)
    sys.exit(1)
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
echo "Writing classifier and run summaries"
echo

RUN_ROOT="${RUN_ROOT}" \
CKPT_PATH="${CHECKPOINT}" \
SLUG="${SLUG}" \
CLF_EPOCHS="${CLF_EPOCHS}" \
CLF_WIDTH="${CLF_WIDTH}" \
CLF_DEPTH="${CLF_DEPTH}" \
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
dataset = "cifar10"
clf_dir = os.path.join(run_root, f"{dataset}_classifier")
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
    f"- **best top-1 accuracy**: {best_acc:.2f}%",
    f"- **best epoch**: {best_epoch}",
    f"- **num epochs trained**: {os.environ['CLF_EPOCHS']}",
    f"- **classifier width**: {os.environ['CLF_WIDTH']}",
    f"- **classifier depth**: {os.environ['CLF_DEPTH']}",
    f"- **classifier dropout**: {os.environ['CLF_DROPOUT']}",
    f"- **classifier lr**: {os.environ['CLF_LR']}",
    f"- **classifier batch size**: {os.environ['CLF_BATCH_SIZE']}",
    f"- **mod_dim (flat input)**: {os.environ['MOD_DIM']}",
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
    "",
    "## Backbone",
    "",
    f"- **checkpoint**: `{ckpt_path}`",
    f"- **best epoch**: {ckpt.get('epoch')}",
    f"- **best loss (mean outer)**: {ckpt.get('loss')}",
    f"- **variant**: `{ckpt.get('variant')}`",
    "",
    "## Classifier",
    "",
    f"- **checkpoint**: `{clf_path}`",
    f"- **best top-1 accuracy**: {best_acc:.2f}%",
    f"- **best epoch**: {best_epoch}",
    f"- **num epochs trained**: {os.environ['CLF_EPOCHS']}",
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
echo "Classifier         : ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo "Classifier summary : ${RUN_ROOT}/cifar10_classifier/checkpoint_summary.md"
echo "Run summary        : ${RUN_ROOT}/run_summary.md"
echo "============================================================"
