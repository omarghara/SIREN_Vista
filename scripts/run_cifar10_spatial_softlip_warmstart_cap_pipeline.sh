#!/bin/bash

# Warm-start CIFAR-10 Spatial-Functa from the trained softlip-tiered checkpoint,
# cap one late layer relative to that same softlip checkpoint, then run:
#   1. 5-epoch meta-training
#   2. inner-5 makeset
#   3. best-sweep CNN classifier
#   4. matched inner-5 PGD on eps 1,2,4,6

set -euo pipefail

TARGET="${TARGET:?set TARGET to readout or pre_readout}"
SCALE="${SCALE:?set SCALE to 0.50 or 0.10}"

if [[ "${TARGET}" != "readout" && "${TARGET}" != "pre_readout" ]]; then
    echo "ERROR: TARGET must be readout or pre_readout, got '${TARGET}'" >&2
    exit 1
fi

CUDA_GPU="${CUDA_GPU:-0}"
DATA_PATH="${DATA_PATH:-../data}"
DEVICE="${DEVICE:-cuda}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXT_LR="${EXT_LR:-3e-05}"
INT_LR="${INT_LR:-0.01}"
INNER_STEPS="${INNER_STEPS:-3}"
INNER_OPTIM="${INNER_OPTIM:-sgd}"
SEED="${SEED:-0}"
LOG_SIGMAS_EVERY="${LOG_SIGMAS_EVERY:-100}"

# Use a strong lambda by default; the previous CIFAR warm-start cap runs with
# lambda=1.0 moved the caps meaningfully, while 1e-2 was too weak.
SPEC_CAP_LAMBDA="${SPEC_CAP_LAMBDA:-1.0}"
SPEC_CAP_POWER_ITERS="${SPEC_CAP_POWER_ITERS:-10}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MAKESET="${RUN_MAKESET:-1}"
RUN_CLASSIFIER="${RUN_CLASSIFIER:-1}"
RUN_PGD="${RUN_PGD:-1}"

MAKE_ITERS="${MAKE_ITERS:-5}"
CLF_EPOCHS="${CLF_EPOCHS:-40}"
PGD_MAX_SAMPLES="${PGD_MAX_SAMPLES:-200}"
PGD_STEPS="${PGD_STEPS:-200}"
PGD_MOD_STEPS="${PGD_MOD_STEPS:-5}"
PGD_LR="${PGD_LR:-0.01}"
PGD_INNER_LR="${PGD_INNER_LR:-0.01}"
EPS_STR="${EPS_STR:-1 2 4 6}"

SOFTLIP_SLUG="functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cifar_spatial_tiered_lam1e-02_sine_and_readout_train50000_test10000"
SOFTLIP_CKPT="${SOFTLIP_CKPT:-model_cifar10/${SOFTLIP_SLUG}/modSiren.pth}"

scale_tag="$(python - <<PYTAG
scale = float("${SCALE}")
print(f"{int(round(scale * 100)):02d}")
PYTAG
)"
target_tag="${TARGET}"
if [[ "${TARGET}" == "pre_readout" ]]; then
    target_tag="prereadout"
fi

MODEL_NAME="${MODEL_NAME:-cifar10_spatial_warmsoftlip_${target_tag}_cap${scale_tag}_lam${SPEC_CAP_LAMBDA}_e${NUM_EPOCHS}}"
MODEL_LABEL="${MODEL_LABEL:-warmsoftlip_${target_tag}_cap${scale_tag}_lam${SPEC_CAP_LAMBDA}_e${NUM_EPOCHS}}"
FUNCTASET_STEM="${FUNCTASET_STEM:-${MODEL_LABEL}_inner${MAKE_ITERS}}"
RUN_ROOT="${RUN_ROOT:-runs/cifar10_spatial_inner5_softlip_warmstart_caps/${MODEL_LABEL}}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${SOFTLIP_CKPT}" ]]; then
    echo "ERROR: missing softlip checkpoint: ${SOFTLIP_CKPT}" >&2
    exit 1
fi

mkdir -p "model_cifar10/${MODEL_NAME}/logs" "${RUN_ROOT}/logs"
TRAIN_LOG="model_cifar10/${MODEL_NAME}/logs/train.log"
SIREN_CKPT="model_cifar10/${MODEL_NAME}/modSiren.pth"

echo "== CIFAR-10 softlip-warmstart cap experiment =="
echo "  target        : ${TARGET}"
echo "  scale         : ${SCALE}"
echo "  lambda        : ${SPEC_CAP_LAMBDA}"
echo "  gpu           : ${CUDA_GPU}"
echo "  softlip ckpt  : ${SOFTLIP_CKPT}"
echo "  model name    : ${MODEL_NAME}"
echo "  model label   : ${MODEL_LABEL}"
echo "  run root      : ${RUN_ROOT}"
echo "  train epochs  : ${NUM_EPOCHS}"
echo "  PGD eps list  : ${EPS_STR}"
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
            --variant spectral_cap \
            --model-name "${MODEL_NAME}" \
            --init-from-checkpoint "${SOFTLIP_CKPT}" \
            --log-sigmas-every "${LOG_SIGMAS_EVERY}" \
            --spec-cap-target "${TARGET}" \
            --spec-cap-mode reference_scale \
            --spec-cap-scale "${SCALE}" \
            --spec-cap-lambda "${SPEC_CAP_LAMBDA}" \
            --spec-cap-reference-checkpoint "${SOFTLIP_CKPT}" \
            --spec-cap-power-iters "${SPEC_CAP_POWER_ITERS}" 2>&1 | tee "${TRAIN_LOG}"
    fi
fi

if [[ ! -f "${SIREN_CKPT}" ]]; then
    echo "ERROR: missing trained checkpoint after train stage: ${SIREN_CKPT}" >&2
    exit 1
fi

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
CLF_EPOCHS="${CLF_EPOCHS}" \
PGD_MAX_SAMPLES="${PGD_MAX_SAMPLES}" \
PGD_STEPS="${PGD_STEPS}" \
PGD_MOD_STEPS="${PGD_MOD_STEPS}" \
PGD_LR="${PGD_LR}" \
PGD_INNER_LR="${PGD_INNER_LR}" \
EPS_STR="${EPS_STR}" \
SEED="${SEED}" \
    bash scripts/run_cifar10_spatial_inner5_checkpoint.sh

cat > "${RUN_ROOT}/softlip_warmstart_cap_config.md" <<EOF_CONFIG
# CIFAR-10 Softlip-Warmstart Cap Experiment

- target: \`${TARGET}\`
- scale relative to softlip checkpoint: \`${SCALE}\`
- spectral-cap lambda: \`${SPEC_CAP_LAMBDA}\`
- train epochs: \`${NUM_EPOCHS}\`
- warm-start checkpoint: \`${SOFTLIP_CKPT}\`
- cap-reference checkpoint: \`${SOFTLIP_CKPT}\`
- trained checkpoint: \`${SIREN_CKPT}\`
- run root: \`${RUN_ROOT}\`
- makeset inner iterations: \`${MAKE_ITERS}\`
- classifier epochs: \`${CLF_EPOCHS}\`
- PGD eps list: \`${EPS_STR}\`
- PGD samples: \`${PGD_MAX_SAMPLES}\`
- PGD steps: \`${PGD_STEPS}\`
- PGD inner phi steps: \`${PGD_MOD_STEPS}\`
EOF_CONFIG

echo
echo "Done softlip-warmstart cap pipeline."
echo "  checkpoint: ${SIREN_CKPT}"
echo "  run root  : ${RUN_ROOT}"
