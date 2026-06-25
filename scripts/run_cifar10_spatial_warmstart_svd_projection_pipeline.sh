#!/bin/bash

# End-to-end CIFAR-10 Spatial-Functa HARD SVD spectral-projection pipeline.
# Warm-starts from the trained vanilla checkpoint, continues meta-training with
# hard SVD projection after every outer step, then runs the standard eval chain:
#   1. meta-training with projection (+ cap verification)
#   2. makeset (functaset fit)
#   3. best-sweep CNN classifier
#   4. spatial PGD attack
#
# Usage:
#   EXPERIMENT=readout_scale          CUDA_GPU=0 SVD_SCALE=0.5 bash scripts/run_cifar10_spatial_warmstart_svd_projection_pipeline.sh
#   EXPERIMENT=all_sine_readout_scale CUDA_GPU=1 SVD_SCALE=0.5 bash scripts/run_cifar10_spatial_warmstart_svd_projection_pipeline.sh

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-${1:-}}"
if [[ -z "${EXPERIMENT}" ]]; then
    echo "ERROR: set EXPERIMENT to one of:" >&2
    echo "  readout_scale, all_sine_readout_scale, all_sine_readout_modul_scale," >&2
    echo "  modul_scale, modul_readout_scale," >&2
    echo "  readout_L, all_sine_readout_L, all_sine_readout_modul_L" >&2
    exit 1
fi

CUDA_GPU="${CUDA_GPU:-0}"
DATA_PATH="${DATA_PATH:-../data}"
DEVICE="${DEVICE:-cuda}"

# --- backbone training ---
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXT_LR="${EXT_LR:-3e-05}"
INT_LR="${INT_LR:-0.01}"
INNER_STEPS="${INNER_STEPS:-3}"
INNER_OPTIM="${INNER_OPTIM:-sgd}"
SEED="${SEED:-0}"
LOG_SIGMAS_EVERY="${LOG_SIGMAS_EVERY:-100}"

# --- projection ---
SVD_CAP="${SVD_CAP:-1.0}"          # effective Lipschitz L (absolute mode)
SVD_SCALE="${SVD_SCALE:-0.5}"      # reference_scale mode
SVD_EVERY="${SVD_EVERY:-1}"

# --- downstream eval (matches inner-3 backbone / notebook eval protocol) ---
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MAKESET="${RUN_MAKESET:-1}"
RUN_CLASSIFIER="${RUN_CLASSIFIER:-1}"
RUN_PGD="${RUN_PGD:-1}"
MAKE_ITERS="${MAKE_ITERS:-3}"
CLF_EPOCHS="${CLF_EPOCHS:-40}"
PGD_MAX_SAMPLES="${PGD_MAX_SAMPLES:-100}"
PGD_STEPS="${PGD_STEPS:-100}"
PGD_MOD_STEPS="${PGD_MOD_STEPS:-3}"
PGD_LR="${PGD_LR:-0.01}"
PGD_INNER_LR="${PGD_INNER_LR:-0.01}"
EPS_STR="${EPS_STR:-1 2 4 6}"

BASE_SLUG="functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000"
BASE_CKPT="${BASE_CKPT:-model_cifar10/${BASE_SLUG}/modSiren.pth}"

VARIANT="vanilla"
MODEL_NAME=""
MODEL_LABEL=""
proj_flags=()

case "${EXPERIMENT}" in
    vanilla_baseline)
        # Matched control: same warm-start + same epochs + same eval, NO projection.
        MODEL_LABEL="warmvanilla_baseline"
        proj_flags=()
        ;;
    readout_scale)
        MODEL_LABEL="svdproj_readout_scale${SVD_SCALE}"
        proj_flags=(--svd-proj --svd-proj-target readout
                    --svd-proj-cap-mode reference_scale --svd-proj-scale "${SVD_SCALE}"
                    --svd-proj-reference-checkpoint "${BASE_CKPT}")
        ;;
    all_sine_readout_scale)
        MODEL_LABEL="svdproj_all_sine_readout_scale${SVD_SCALE}"
        proj_flags=(--svd-proj --svd-proj-target all_sine_readout
                    --svd-proj-cap-mode reference_scale --svd-proj-scale "${SVD_SCALE}"
                    --svd-proj-reference-checkpoint "${BASE_CKPT}")
        ;;
    all_sine_readout_modul_scale)
        MODEL_LABEL="svdproj_all_sine_readout_modul_scale${SVD_SCALE}"
        proj_flags=(--svd-proj --svd-proj-target all_sine_readout_modul
                    --svd-proj-cap-mode reference_scale --svd-proj-scale "${SVD_SCALE}"
                    --svd-proj-reference-checkpoint "${BASE_CKPT}")
        ;;
    modul_scale)
        MODEL_LABEL="svdproj_modul_scale${SVD_SCALE}"
        proj_flags=(--svd-proj --svd-proj-target modul
                    --svd-proj-cap-mode reference_scale --svd-proj-scale "${SVD_SCALE}"
                    --svd-proj-reference-checkpoint "${BASE_CKPT}")
        ;;
    modul_readout_scale)
        MODEL_LABEL="svdproj_modul_readout_scale${SVD_SCALE}"
        proj_flags=(--svd-proj --svd-proj-target modul_readout
                    --svd-proj-cap-mode reference_scale --svd-proj-scale "${SVD_SCALE}"
                    --svd-proj-reference-checkpoint "${BASE_CKPT}")
        ;;
    readout_L)
        MODEL_LABEL="svdproj_readout_L${SVD_CAP}"
        proj_flags=(--svd-proj --svd-proj-target readout
                    --svd-proj-cap-mode absolute --svd-proj-cap "${SVD_CAP}")
        ;;
    all_sine_readout_L)
        MODEL_LABEL="svdproj_all_sine_readout_L${SVD_CAP}"
        proj_flags=(--svd-proj --svd-proj-target all_sine_readout
                    --svd-proj-cap-mode absolute --svd-proj-cap "${SVD_CAP}")
        ;;
    all_sine_readout_modul_L)
        MODEL_LABEL="svdproj_all_sine_readout_modul_L${SVD_CAP}"
        proj_flags=(--svd-proj --svd-proj-target all_sine_readout_modul
                    --svd-proj-cap-mode absolute --svd-proj-cap "${SVD_CAP}")
        ;;
    *)
        echo "ERROR: unknown EXPERIMENT='${EXPERIMENT}'" >&2
        exit 1
        ;;
esac

if [[ ${#proj_flags[@]} -gt 0 ]]; then
    proj_flags+=(--svd-proj-every "${SVD_EVERY}")
fi

MODEL_LABEL="${MODEL_LABEL}_e${NUM_EPOCHS}"
MODEL_NAME="${MODEL_NAME:-cifar10_spatial_warmvanilla_${MODEL_LABEL}}"
FUNCTASET_STEM="${FUNCTASET_STEM:-${MODEL_LABEL}_inner${MAKE_ITERS}}"
RUN_ROOT="${RUN_ROOT:-runs/cifar10_spatial_svd_projection/${MODEL_LABEL}}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "ERROR: missing base checkpoint: ${BASE_CKPT}" >&2
    exit 1
fi

mkdir -p "model_cifar10/${MODEL_NAME}/logs" "${RUN_ROOT}/logs"
TRAIN_LOG="model_cifar10/${MODEL_NAME}/logs/train.log"
SIREN_CKPT="model_cifar10/${MODEL_NAME}/modSiren.pth"

echo "== CIFAR-10 hard SVD-projection pipeline =="
echo "  experiment    : ${EXPERIMENT}"
echo "  gpu           : ${CUDA_GPU}"
echo "  base ckpt     : ${BASE_CKPT}"
echo "  model name    : ${MODEL_NAME}"
echo "  proj flags    : ${proj_flags[*]}"
echo "  train epochs  : ${NUM_EPOCHS}"
echo "  eval inner    : makeset=${MAKE_ITERS}, pgd mod-steps=${PGD_MOD_STEPS}"
echo "  pgd           : steps=${PGD_STEPS}, n=${PGD_MAX_SAMPLES}, eps=${EPS_STR}"
echo "  run root      : ${RUN_ROOT}"
echo

if [[ "${RUN_TRAIN}" == "1" ]]; then
    if [[ -f "${SIREN_CKPT}" ]]; then
        echo "[skip train] found ${SIREN_CKPT}"
    else
        python -u trainer.py \
            --dataset cifar10 \
            --data-path "${DATA_PATH}" \
            --device "${DEVICE}" \
            --seed "${SEED}" \
            --ext-lr "${EXT_LR}" \
            --int-lr "${INT_LR}" \
            --batch-size "${BATCH_SIZE}" \
            --num-epochs "${NUM_EPOCHS}" \
            --inner-steps "${INNER_STEPS}" \
            --inner-optim "${INNER_OPTIM}" \
            --hidden-dim 256 \
            --mod-dim 1024 \
            --depth 6 \
            --inr-type siren \
            --freq 10.0 \
            --spatial-modulation \
            --latent-spatial-dim 8 \
            --latent-dim 16 \
            --spatial-interp nearest \
            --use-local-coords \
            --modulation-type shift \
            --variant "${VARIANT}" \
            --model-name "${MODEL_NAME}" \
            --init-from-checkpoint "${BASE_CKPT}" \
            --log-sigmas-every "${LOG_SIGMAS_EVERY}" \
            ${proj_flags[@]+"${proj_flags[@]}"} 2>&1 | tee "${TRAIN_LOG}"
    fi
fi

if [[ ! -f "${SIREN_CKPT}" ]]; then
    echo "ERROR: missing trained checkpoint after train stage: ${SIREN_CKPT}" >&2
    exit 1
fi

echo
echo "== Verifying enforced spectral caps on the saved checkpoint =="
python scripts/verify_svd_projection.py --checkpoint "${SIREN_CKPT}" || true

# Downstream eval chain: makeset -> classifier -> PGD (vanilla architecture).
MODEL_LABEL="${MODEL_LABEL}" \
SIREN_CKPT="${SIREN_CKPT}" \
FUNCTASET_STEM="${FUNCTASET_STEM}" \
RUN_ROOT="${RUN_ROOT}" \
RUN_MAKESET="${RUN_MAKESET}" \
RUN_CLASSIFIER="${RUN_CLASSIFIER}" \
RUN_PGD="${RUN_PGD}" \
CUDA_GPU="${CUDA_GPU}" \
DATA_PATH="${DATA_PATH}" \
MAKE_ITERS="${MAKE_ITERS}" \
MAKESET_VARIANT="vanilla" \
CLF_EPOCHS="${CLF_EPOCHS}" \
PGD_MAX_SAMPLES="${PGD_MAX_SAMPLES}" \
PGD_STEPS="${PGD_STEPS}" \
PGD_MOD_STEPS="${PGD_MOD_STEPS}" \
PGD_LR="${PGD_LR}" \
PGD_INNER_LR="${PGD_INNER_LR}" \
EPS_STR="${EPS_STR}" \
SEED="${SEED}" \
    bash scripts/run_cifar10_spatial_inner5_checkpoint.sh

echo
echo "Done SVD-projection pipeline."
echo "  checkpoint : ${SIREN_CKPT}"
echo "  run root   : ${RUN_ROOT}"
echo "  summary    : ${RUN_ROOT}/pgd_inner${PGD_MOD_STEPS}_summary.md"
