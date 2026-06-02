#!/bin/bash

# Warm-start CIFAR-10 Spatial-Functa meta-learning from the trained vanilla
# checkpoint, then optimize a new regularizer variant with a fresh optimizer.
#
# Usage examples:
#   EXPERIMENT=orthogonal CUDA_GPU=0 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
#   EXPERIMENT=readout_cap90 CUDA_GPU=0 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
#   EXPERIMENT=readout_cap50 CUDA_GPU=1 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
#   EXPERIMENT=readout_cap10 CUDA_GPU=0 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
#   EXPERIMENT=pre_readout_cap10 CUDA_GPU=1 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
#   EXPERIMENT=readout_counter CUDA_GPU=0 COUNTER_TARGET=1 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
#   EXPERIMENT=pre_readout_counter CUDA_GPU=1 COUNTER_TARGET=1 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-${1:-}}"
if [[ -z "${EXPERIMENT}" ]]; then
    echo "ERROR: set EXPERIMENT to one of:" >&2
    echo "  orthogonal, readout_cap90, readout_cap50, readout_cap10," >&2
    echo "  pre_readout_cap10, readout_counter, pre_readout_counter" >&2
    exit 1
fi

CUDA_GPU="${CUDA_GPU:-0}"
DATA_PATH="${DATA_PATH:-../data}"
DEVICE="${DEVICE:-cuda}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXT_LR="${EXT_LR:-3e-05}"
INT_LR="${INT_LR:-0.01}"
INNER_STEPS="${INNER_STEPS:-3}"
INNER_OPTIM="${INNER_OPTIM:-sgd}"
SEED="${SEED:-0}"
LOG_SIGMAS_EVERY="${LOG_SIGMAS_EVERY:-100}"

BASE_SLUG="functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000"
BASE_CKPT="${BASE_CKPT:-model_cifar10/${BASE_SLUG}/modSiren.pth}"

ORTH_LAMBDA="${ORTH_LAMBDA:-1e-3}"
ORTH_APPLY_TO="${ORTH_APPLY_TO:-sine_and_readout}"
ORTH_FORM="${ORTH_FORM:-auto}"

SPEC_CAP_LAMBDA="${SPEC_CAP_LAMBDA:-1e-2}"
SPEC_CAP_POWER_ITERS="${SPEC_CAP_POWER_ITERS:-10}"
COUNTER_TARGET="${COUNTER_TARGET:-1.0}"

VARIANT=""
MODEL_NAME=""
variant_flags=()

case "${EXPERIMENT}" in
    orthogonal)
        VARIANT="orthogonal"
        MODEL_NAME="cifar10_spatial_warmvanilla_orth_${ORTH_APPLY_TO}_${ORTH_FORM}_lam${ORTH_LAMBDA}"
        variant_flags=(
            --orth-lambda "${ORTH_LAMBDA}"
            --orth-apply-to "${ORTH_APPLY_TO}"
            --orth-form "${ORTH_FORM}"
        )
        ;;
    readout_cap90)
        VARIANT="spectral_cap"
        MODEL_NAME="cifar10_spatial_warmvanilla_readout_cap90_lam${SPEC_CAP_LAMBDA}"
        variant_flags=(
            --spec-cap-target readout
            --spec-cap-mode reference_scale
            --spec-cap-scale 0.90
            --spec-cap-lambda "${SPEC_CAP_LAMBDA}"
            --spec-cap-reference-checkpoint "${BASE_CKPT}"
            --spec-cap-power-iters "${SPEC_CAP_POWER_ITERS}"
        )
        ;;
    readout_cap50)
        VARIANT="spectral_cap"
        MODEL_NAME="cifar10_spatial_warmvanilla_readout_cap50_lam${SPEC_CAP_LAMBDA}"
        variant_flags=(
            --spec-cap-target readout
            --spec-cap-mode reference_scale
            --spec-cap-scale 0.50
            --spec-cap-lambda "${SPEC_CAP_LAMBDA}"
            --spec-cap-reference-checkpoint "${BASE_CKPT}"
            --spec-cap-power-iters "${SPEC_CAP_POWER_ITERS}"
        )
        ;;
    readout_cap10)
        VARIANT="spectral_cap"
        MODEL_NAME="cifar10_spatial_warmvanilla_readout_cap10_lam${SPEC_CAP_LAMBDA}"
        variant_flags=(
            --spec-cap-target readout
            --spec-cap-mode reference_scale
            --spec-cap-scale 0.10
            --spec-cap-lambda "${SPEC_CAP_LAMBDA}"
            --spec-cap-reference-checkpoint "${BASE_CKPT}"
            --spec-cap-power-iters "${SPEC_CAP_POWER_ITERS}"
        )
        ;;
    pre_readout_cap10)
        VARIANT="spectral_cap"
        MODEL_NAME="cifar10_spatial_warmvanilla_prereadout_cap10_lam${SPEC_CAP_LAMBDA}"
        variant_flags=(
            --spec-cap-target pre_readout
            --spec-cap-mode reference_scale
            --spec-cap-scale 0.10
            --spec-cap-lambda "${SPEC_CAP_LAMBDA}"
            --spec-cap-reference-checkpoint "${BASE_CKPT}"
            --spec-cap-power-iters "${SPEC_CAP_POWER_ITERS}"
        )
        ;;
    readout_counter)
        VARIANT="spectral_cap"
        MODEL_NAME="cifar10_spatial_warmvanilla_readout_counter${COUNTER_TARGET}_lam${SPEC_CAP_LAMBDA}"
        variant_flags=(
            --spec-cap-target readout
            --spec-cap-mode counter_amplification
            --spec-cap-counter-target "${COUNTER_TARGET}"
            --spec-cap-lambda "${SPEC_CAP_LAMBDA}"
            --spec-cap-reference-checkpoint "${BASE_CKPT}"
            --spec-cap-power-iters "${SPEC_CAP_POWER_ITERS}"
        )
        ;;
    pre_readout_counter)
        VARIANT="spectral_cap"
        MODEL_NAME="cifar10_spatial_warmvanilla_prereadout_counter${COUNTER_TARGET}_lam${SPEC_CAP_LAMBDA}"
        variant_flags=(
            --spec-cap-target pre_readout
            --spec-cap-mode counter_amplification
            --spec-cap-counter-target "${COUNTER_TARGET}"
            --spec-cap-lambda "${SPEC_CAP_LAMBDA}"
            --spec-cap-reference-checkpoint "${BASE_CKPT}"
            --spec-cap-power-iters "${SPEC_CAP_POWER_ITERS}"
        )
        ;;
    *)
        echo "ERROR: unknown EXPERIMENT='${EXPERIMENT}'" >&2
        exit 1
        ;;
esac

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "ERROR: missing base checkpoint: ${BASE_CKPT}" >&2
    exit 1
fi

mkdir -p "model_cifar10/${MODEL_NAME}/logs"
LOG_PATH="model_cifar10/${MODEL_NAME}/logs/train.log"

echo "== Warm-start CIFAR-10 Spatial-Functa regularizer experiment =="
echo "  experiment    : ${EXPERIMENT}"
echo "  gpu           : ${CUDA_GPU}"
echo "  base ckpt     : ${BASE_CKPT}"
echo "  variant       : ${VARIANT}"
echo "  model name    : ${MODEL_NAME}"
echo "  epochs        : ${NUM_EPOCHS}"
echo "  ext/int lr    : ${EXT_LR} / ${INT_LR}"
echo "  inner         : ${INNER_STEPS} steps, ${INNER_OPTIM}"
echo "  log           : ${LOG_PATH}"
echo

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
    "${variant_flags[@]}" 2>&1 | tee "${LOG_PATH}"
