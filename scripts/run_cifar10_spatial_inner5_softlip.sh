#!/bin/bash

# CIFAR-10 Spatial-Functa soft-Lipschitz tiered inner-5 pipeline.
#
# This script creates a fresh 5-step functaset, trains the best-sweep CNN
# classifier on it, and runs Full-PGD with the same 5-step phi fitting budget.

set -euo pipefail

MODEL_LABEL="softlip_tiered_e12"
FUNCTASET_STEM="${FUNCTASET_STEM:-softlip_tiered_e12_inner5}"
RUN_ROOT="${RUN_ROOT:-runs/cifar10_spatial_inner5_make5_clfbest_v1/${MODEL_LABEL}}"

MODEL_SLUG="functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cifar_spatial_tiered_lam1e-02_sine_and_readout_train50000_test10000"
SIREN_CKPT="${SIREN_CKPT:-model_cifar10/${MODEL_SLUG}/modSiren.pth}"

RUN_MAKESET="${RUN_MAKESET:-1}"
RUN_CLASSIFIER="${RUN_CLASSIFIER:-1}"
RUN_PGD="${RUN_PGD:-1}"

CUDA_GPU="${CUDA_GPU:-1}"
DATA_PATH="${DATA_PATH:-../data}"
MAKE_DEVICE="${MAKE_DEVICE:-cuda}"
CLF_DEVICE="${CLF_DEVICE:-cuda}"
ATTACK_DEVICE="${ATTACK_DEVICE:-cuda}"

MAKE_ITERS="${MAKE_ITERS:-5}"
MAKE_LR="${MAKE_LR:-0.01}"
MAKE_INNER_OPTIM="${MAKE_INNER_OPTIM:-sgd}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-50000}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-10000}"

CLF_LR="${CLF_LR:-0.003}"
CLF_DROPOUT="${CLF_DROPOUT:-0.1}"
CLF_CNN_WIDTH="${CLF_CNN_WIDTH:-256}"
CLF_BATCH_SIZE="${CLF_BATCH_SIZE:-256}"
CLF_WEIGHT_DECAY="${CLF_WEIGHT_DECAY:-0.0}"
CLF_LABEL_SMOOTHING="${CLF_LABEL_SMOOTHING:-0.0}"
CLF_EPOCHS="${CLF_EPOCHS:-40}"
CLF_NORMALIZE_PHI="${CLF_NORMALIZE_PHI:-1}"
LATENT_SPATIAL_DIM="${LATENT_SPATIAL_DIM:-8}"
LATENT_DIM="${LATENT_DIM:-16}"
MOD_DIM="${MOD_DIM:-1024}"

SOFT_LIP_CAP="${SOFT_LIP_CAP:-1.0}"
SOFT_LIP_LAMBDA="${SOFT_LIP_LAMBDA:-0.01}"
SOFT_LIP_APPLY_TO="${SOFT_LIP_APPLY_TO:-sine_and_readout}"

PGD_MOD_STEPS="${PGD_MOD_STEPS:-5}"
PGD_INNER_LR="${PGD_INNER_LR:-0.01}"
PGD_STEPS="${PGD_STEPS:-200}"
PGD_LR="${PGD_LR:-0.01}"
PGD_MAX_SAMPLES="${PGD_MAX_SAMPLES:-200}"
EPS_STR="${EPS_STR:-8}"
read -r -a EPS_LIST <<< "${EPS_STR}"
SEED="${SEED:-0}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_ALL="${RUN_ROOT}/functaset/${FUNCTASET_STEM}_train_all${MAX_TRAIN_SAMPLES}.pkl"
TEST_PKL="${RUN_ROOT}/functaset/${FUNCTASET_STEM}_test.pkl"
CLF_DIR="${RUN_ROOT}/cifar10_cnn_classifier_best_sweep_inner${MAKE_ITERS}"
CLF_CKPT="${CLF_DIR}/best_classifier.pth"
PGD_DIR="${RUN_ROOT}/pgd_cifar10_spatial_cnn_inner${PGD_MOD_STEPS}"

mkdir -p "${RUN_ROOT}/logs" "${CLF_DIR}" "${PGD_DIR}"

if [[ ! -f "${SIREN_CKPT}" ]]; then
    echo "ERROR: missing SIREN checkpoint: ${SIREN_CKPT}" >&2
    exit 1
fi

echo "== CIFAR-10 soft-Lipschitz tiered inner-${MAKE_ITERS} pipeline =="
echo "  run root      : ${RUN_ROOT}"
echo "  checkpoint    : ${SIREN_CKPT}"
echo "  data path     : ${DATA_PATH}"
echo "  make fit      : steps=${MAKE_ITERS}, lr=${MAKE_LR}, optim=${MAKE_INNER_OPTIM}"
echo "  soft lip      : cap=${SOFT_LIP_CAP}, lambda=${SOFT_LIP_LAMBDA}, apply_to=${SOFT_LIP_APPLY_TO}"
echo "  classifier    : cnn width=${CLF_CNN_WIDTH}, lr=${CLF_LR}, dropout=${CLF_DROPOUT}, normalize_phi=${CLF_NORMALIZE_PHI}, epochs=${CLF_EPOCHS}"
echo "  pgd           : eps=${EPS_LIST[*]} /255, n=${PGD_MAX_SAMPLES}, pgd_steps=${PGD_STEPS}, mod_steps=${PGD_MOD_STEPS}"
echo

python - <<'PYCUDA'
import torch
print("cuda available:", torch.cuda.is_available())
try:
    print("visible gpus:", torch.cuda.device_count())
    if torch.cuda.device_count():
        print("gpu name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("cuda probe warning:", repr(exc))
PYCUDA

if [[ "${RUN_MAKESET}" == "1" ]]; then
    MAKE_LOG="${RUN_ROOT}/logs/makeset_inner${MAKE_ITERS}.log"
    echo
    echo "---- makeset ${MODEL_LABEL}: ${MAKE_ITERS} inner steps ----"
    echo "     train_all : ${TRAIN_ALL}"
    echo "     test      : ${TEST_PKL}"
    echo "     log       : ${MAKE_LOG}"

    if [[ -f "${TRAIN_ALL}" && -f "${TEST_PKL}" ]]; then
        echo "     [skip] found existing functaset files"
    else
        python -u makeset.py \
            --dataset cifar10 \
            --data-path "${DATA_PATH}" \
            --checkpoint "${SIREN_CKPT}" \
            --saveroot "${RUN_ROOT}" \
            --functaset-stem "${FUNCTASET_STEM}" \
            --iters "${MAKE_ITERS}" \
            --lr "${MAKE_LR}" \
            --inner-optim "${MAKE_INNER_OPTIM}" \
            --max-train-samples "${MAX_TRAIN_SAMPLES}" \
            --max-test-samples "${MAX_TEST_SAMPLES}" \
            --save-train-all \
            --device "${MAKE_DEVICE}" \
            --variant soft_lipschitz \
            --soft-lip-cap "${SOFT_LIP_CAP}" \
            --soft-lip-lambda "${SOFT_LIP_LAMBDA}" \
            --soft-lip-apply-to "${SOFT_LIP_APPLY_TO}" 2>&1 | tee "${MAKE_LOG}"
    fi
fi

if [[ "${RUN_CLASSIFIER}" == "1" ]]; then
    CLF_LOG="${RUN_ROOT}/logs/train_classifier_best_sweep.log"
    echo
    echo "---- train classifier ${MODEL_LABEL}: best sweep CNN params ----"
    echo "     train : ${TRAIN_ALL}"
    echo "     test  : ${TEST_PKL}"
    echo "     out   : ${CLF_DIR}"
    echo "     log   : ${CLF_LOG}"

    if [[ -f "${CLF_CKPT}" ]]; then
        echo "     [skip] found ${CLF_CKPT}"
    else
        if [[ ! -f "${TRAIN_ALL}" || ! -f "${TEST_PKL}" ]]; then
            echo "ERROR: missing functaset files; run makeset first." >&2
            exit 1
        fi

        norm_flag=()
        if [[ "${CLF_NORMALIZE_PHI}" == "1" ]]; then
            norm_flag=(--normalize-phi)
        fi

        python -u train_classifier.py \
            --classifier-type cnn \
            --dataset cifar10 \
            --functaset-path-train "${TRAIN_ALL}" \
            --functaset-path-test "${TEST_PKL}" \
            --save-dir "${CLF_DIR}" \
            --device "${CLF_DEVICE}" \
            --latent-spatial-dim "${LATENT_SPATIAL_DIM}" \
            --latent-dim "${LATENT_DIM}" \
            --mod-dim "${MOD_DIM}" \
            --lr "${CLF_LR}" \
            --weight-decay "${CLF_WEIGHT_DECAY}" \
            --label-smoothing "${CLF_LABEL_SMOOTHING}" \
            --cnn-width "${CLF_CNN_WIDTH}" \
            --dropout "${CLF_DROPOUT}" \
            --batch-size "${CLF_BATCH_SIZE}" \
            --num-epochs "${CLF_EPOCHS}" \
            "${norm_flag[@]}" 2>&1 | tee "${CLF_LOG}"
    fi
fi

if [[ "${RUN_PGD}" == "1" ]]; then
    if [[ ! -f "${CLF_CKPT}" ]]; then
        echo "ERROR: missing classifier checkpoint: ${CLF_CKPT}" >&2
        exit 1
    fi

    for eps in "${EPS_LIST[@]}"; do
        PGD_STEM="${PGD_DIR}/eps${eps}_n${PGD_MAX_SAMPLES}"
        PGD_JSON="${PGD_STEM}.json"
        PGD_LOG="${PGD_STEM}.log"

        echo
        echo "---- PGD ${MODEL_LABEL}: eps=${eps}/255, n=${PGD_MAX_SAMPLES}, mod_steps=${PGD_MOD_STEPS} ----"
        echo "     json : ${PGD_JSON}"
        echo "     log  : ${PGD_LOG}"

        if [[ -f "${PGD_JSON}" ]]; then
            echo "     [skip] found ${PGD_JSON}"
            continue
        fi

        python -u attacks/full_pgd_cifar10_spatial.py \
            --data-path "${DATA_PATH}" \
            --siren-checkpoint "${SIREN_CKPT}" \
            --classifier-checkpoint "${CLF_CKPT}" \
            --epsilon "${eps}" \
            --pgd-steps "${PGD_STEPS}" \
            --mod-steps "${PGD_MOD_STEPS}" \
            --ext-lr "${PGD_LR}" \
            --inner-lr "${PGD_INNER_LR}" \
            --seed "${SEED}" \
            --max-samples "${PGD_MAX_SAMPLES}" \
            --output-json "${PGD_JSON}" \
            --device "${ATTACK_DEVICE}" 2>&1 | tee "${PGD_LOG}"
    done
fi

SUMMARY_JSON="${RUN_ROOT}/pgd_inner${PGD_MOD_STEPS}_summary.json"
SUMMARY_MD="${RUN_ROOT}/pgd_inner${PGD_MOD_STEPS}_summary.md"
python - <<PY_SUMMARY
import glob
import json
import os

records = []
for path in sorted(glob.glob(os.path.join("${PGD_DIR}", "eps*_n*.json"))):
    with open(path) as f:
        rec = json.load(f)
    rec["model"] = "${MODEL_LABEL}"
    rec["path"] = path
    rec["epsilon_255"] = int(round(rec["constraint"] * 255))
    records.append(rec)

records.sort(key=lambda r: r["epsilon_255"])
with open("${SUMMARY_JSON}", "w") as f:
    json.dump(records, f, indent=2)

lines = [
    "# CIFAR-10 Spatial-Functa PGD Summary - softlip tiered inner5",
    "",
    "Run root: ${RUN_ROOT}",
    "",
    "Attack mod steps: ${PGD_MOD_STEPS}",
    "PGD steps: ${PGD_STEPS}",
    "",
    "| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in records:
    gap = r["clean_acc"] - r["robust_acc"]
    lines.append(
        f"| {r['model']} | {r['epsilon_255']} | {r['n_samples']} | "
        f"{r['clean_acc']:.4f} | {r['robust_acc']:.4f} | "
        f"{r['conditional_robust_acc']:.4f} | {gap:+.4f} |"
    )
with open("${SUMMARY_MD}", "w") as f:
    f.write("\\n".join(lines) + "\\n")
if records:
    print("\\n".join(lines))
else:
    print("[summary] no PGD json files found yet")
print("[summary] JSON:", "${SUMMARY_JSON}")
print("[summary] MD  :", "${SUMMARY_MD}")
PY_SUMMARY

cat > "${RUN_ROOT}/run_config.md" <<EOF_CONFIG
# CIFAR-10 Soft-Lipschitz Tiered Spatial-Functa Inner-5 Run Config

- SIREN checkpoint: \`${SIREN_CKPT}\`
- functaset train_all: \`${TRAIN_ALL}\`
- functaset test: \`${TEST_PKL}\`
- makeset inner steps: \`${MAKE_ITERS}\`
- makeset inner lr: \`${MAKE_LR}\`
- makeset optimizer: \`${MAKE_INNER_OPTIM}\`
- soft-lip cap: \`${SOFT_LIP_CAP}\`
- soft-lip lambda: \`${SOFT_LIP_LAMBDA}\`
- soft-lip apply to: \`${SOFT_LIP_APPLY_TO}\`
- classifier type: \`cnn\`
- classifier lr: \`${CLF_LR}\`
- classifier width: \`${CLF_CNN_WIDTH}\`
- classifier dropout: \`${CLF_DROPOUT}\`
- classifier normalize phi: \`${CLF_NORMALIZE_PHI}\`
- classifier epochs: \`${CLF_EPOCHS}\`
- classifier checkpoint: \`${CLF_CKPT}\`
- PGD mod steps: \`${PGD_MOD_STEPS}\`
- PGD steps: \`${PGD_STEPS}\`
- PGD LR: \`${PGD_LR}\`
- PGD eps list: \`${EPS_STR}\`
- PGD max samples: \`${PGD_MAX_SAMPLES}\`
EOF_CONFIG

echo
echo "Done."
echo "  Run root : ${RUN_ROOT}"
echo "  Config   : ${RUN_ROOT}/run_config.md"
echo "  Summary  : ${SUMMARY_MD}"
