#!/bin/bash

# CIFAR-10 Spatial-Functa inner-5 pipeline.
#
# Purpose:
#   1. Refit vanilla and soft-Lipschitz tiered CIFAR-10 spatial functasets
#      with 5 inner phi iterations.
#   2. Train new CNN classifiers using the best sweep parameters found for
#      the spatial-phi CNN.
#   3. Run Full-PGD on 200 test images with the attack also using 5 inner phi
#      iterations, removing the previous train/attack fitting mismatch.
#
# Outputs are placed under a fresh experiment root and do not overwrite the
# original functasets or classifiers.

set -euo pipefail

# ---- experiment roots --------------------------------------------------------

EXP_ROOT="${EXP_ROOT:-runs/cifar10_spatial_inner5_make5_clfbest_v1}"

VANILLA_LABEL="vanilla_e512"
SOFTLIP_LABEL="softlip_tiered_e12"

VANILLA_RUN="${EXP_ROOT}/${VANILLA_LABEL}"
SOFTLIP_RUN="${EXP_ROOT}/${SOFTLIP_LABEL}"

VANILLA_STEM="${VANILLA_LABEL}_inner5"
SOFTLIP_STEM="${SOFTLIP_LABEL}_inner5"

# ---- source checkpoints ------------------------------------------------------

VANILLA_SLUG="functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000"
SOFTLIP_SLUG="functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cifar_spatial_tiered_lam1e-02_sine_and_readout_train50000_test10000"

VANILLA_CKPT="model_cifar10/${VANILLA_SLUG}/modSiren.pth"
SOFTLIP_CKPT="model_cifar10/${SOFTLIP_SLUG}/modSiren.pth"

# ---- stage toggles -----------------------------------------------------------

RUN_VANILLA="${RUN_VANILLA:-1}"
RUN_SOFTLIP="${RUN_SOFTLIP:-1}"
RUN_MAKESET="${RUN_MAKESET:-1}"
RUN_CLASSIFIER="${RUN_CLASSIFIER:-1}"
RUN_PGD="${RUN_PGD:-1}"

# ---- compute / data ----------------------------------------------------------

CUDA_GPU="${CUDA_GPU:-0}"
ATTACK_DEVICE="${ATTACK_DEVICE:-cuda}"
DATA_PATH="${DATA_PATH:-../data}"

# ---- matched phi fitting budget ---------------------------------------------

MAKE_ITERS="${MAKE_ITERS:-5}"
MAKE_LR="${MAKE_LR:-0.01}"
MAKE_INNER_OPTIM="${MAKE_INNER_OPTIM:-sgd}"

PGD_MOD_STEPS="${PGD_MOD_STEPS:-5}"
PGD_INNER_LR="${PGD_INNER_LR:-0.01}"

# Full CIFAR-10 by default. Lower these for a quick smoke run.
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-50000}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-10000}"

# ---- best CNN classifier sweep parameters -----------------------------------

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

# ---- PGD knobs ---------------------------------------------------------------

PGD_STEPS="${PGD_STEPS:-200}"
PGD_LR="${PGD_LR:-0.01}"
PGD_MAX_SAMPLES="${PGD_MAX_SAMPLES:-200}"

# Use EPS_STR="1 2 4 6 8" for a small-epsilon sweep. Default is a single
# eps=8/255 run because the full pipeline is already expensive.
EPS_STR="${EPS_STR:-8}"
read -r -a EPS_LIST <<< "${EPS_STR}"

SEED="${SEED:-0}"

# ---- environment -------------------------------------------------------------

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

mkdir -p "${EXP_ROOT}" "${VANILLA_RUN}" "${SOFTLIP_RUN}"

echo "== CIFAR-10 Spatial-Functa inner-5 pipeline =="
echo "  exp root          : ${EXP_ROOT}"
echo "  data path         : ${DATA_PATH}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  make iters/lr     : ${MAKE_ITERS} / ${MAKE_LR} (${MAKE_INNER_OPTIM})"
echo "  classifier params : lr=${CLF_LR}, width=${CLF_CNN_WIDTH}, dropout=${CLF_DROPOUT}, normalize=${CLF_NORMALIZE_PHI}, epochs=${CLF_EPOCHS}"
echo "  PGD               : eps=${EPS_LIST[*]} /255, n=${PGD_MAX_SAMPLES}, steps=${PGD_STEPS}, mod_steps=${PGD_MOD_STEPS}"
echo

python - <<'PYCUDA'
import torch
print('cuda available:', torch.cuda.is_available())
try:
    n = torch.cuda.device_count()
    print('visible gpus:', n)
    if n:
        print('gpu name:', torch.cuda.get_device_name(0))
except Exception as exc:
    print('cuda probe warning:', repr(exc))
PYCUDA

for ck in "${VANILLA_CKPT}" "${SOFTLIP_CKPT}"; do
    if [[ ! -f "${ck}" ]]; then
        echo "ERROR: missing checkpoint ${ck}" >&2
        exit 1
    fi
done

make_functaset() {
    local label=$1
    local ckpt=$2
    local run_root=$3
    local stem=$4
    local variant=$5

    local train_all="${run_root}/functaset/${stem}_train_all${MAX_TRAIN_SAMPLES}.pkl"
    local test_pkl="${run_root}/functaset/${stem}_test.pkl"
    local log="${run_root}/logs/makeset_inner${MAKE_ITERS}.log"

    mkdir -p "${run_root}/logs"

    echo
    echo "---- makeset ${label}: ${MAKE_ITERS} inner steps ----"
    echo "     checkpoint : ${ckpt}"
    echo "     output     : ${run_root}/functaset"
    echo "     log        : ${log}"

    if [[ -f "${train_all}" && -f "${test_pkl}" ]]; then
        echo "     [skip] found ${train_all} and ${test_pkl}"
        return 0
    fi

    local variant_flags=(--variant "${variant}")
    if [[ "${variant}" == "soft_lipschitz" ]]; then
        variant_flags+=(
            --soft-lip-cap 1.0
            --soft-lip-lambda 0.01
            --soft-lip-apply-to sine_and_readout
        )
    fi

    python -u makeset.py \
        --dataset cifar10 \
        --data-path "${DATA_PATH}" \
        --checkpoint "${ckpt}" \
        --saveroot "${run_root}" \
        --functaset-stem "${stem}" \
        --iters "${MAKE_ITERS}" \
        --lr "${MAKE_LR}" \
        --inner-optim "${MAKE_INNER_OPTIM}" \
        --max-train-samples "${MAX_TRAIN_SAMPLES}" \
        --max-test-samples "${MAX_TEST_SAMPLES}" \
        --save-train-all \
        --device cuda \
        "${variant_flags[@]}" 2>&1 | tee "${log}"
}

train_cnn_classifier() {
    local label=$1
    local run_root=$2
    local stem=$3

    local train_all="${run_root}/functaset/${stem}_train_all${MAX_TRAIN_SAMPLES}.pkl"
    local test_pkl="${run_root}/functaset/${stem}_test.pkl"
    local clf_dir="${run_root}/cifar10_cnn_classifier_best_sweep_inner${MAKE_ITERS}"
    local clf_ckpt="${clf_dir}/best_classifier.pth"
    local log="${run_root}/logs/train_classifier_best_sweep.log"

    mkdir -p "${run_root}/logs" "${clf_dir}"

    echo
    echo "---- train classifier ${label}: best sweep CNN params ----"
    echo "     train : ${train_all}"
    echo "     test  : ${test_pkl}"
    echo "     out   : ${clf_dir}"
    echo "     log   : ${log}"

    if [[ -f "${clf_ckpt}" ]]; then
        echo "     [skip] found ${clf_ckpt}"
        return 0
    fi

    if [[ ! -f "${train_all}" || ! -f "${test_pkl}" ]]; then
        echo "ERROR: missing functaset for ${label}; run makeset first." >&2
        exit 1
    fi

    local norm_flag=()
    if [[ "${CLF_NORMALIZE_PHI}" == "1" ]]; then
        norm_flag=(--normalize-phi)
    fi

    python -u train_classifier.py \
        --classifier-type cnn \
        --dataset cifar10 \
        --functaset-path-train "${train_all}" \
        --functaset-path-test "${test_pkl}" \
        --save-dir "${clf_dir}" \
        --device cuda \
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
        "${norm_flag[@]}" 2>&1 | tee "${log}"
}

run_pgd() {
    local label=$1
    local ckpt=$2
    local run_root=$3

    local clf_ckpt="${run_root}/cifar10_cnn_classifier_best_sweep_inner${MAKE_ITERS}/best_classifier.pth"
    local pgd_dir="${run_root}/pgd_cifar10_spatial_cnn_inner${PGD_MOD_STEPS}"
    mkdir -p "${pgd_dir}" "${run_root}/logs"

    if [[ ! -f "${clf_ckpt}" ]]; then
        echo "ERROR: missing classifier checkpoint for ${label}: ${clf_ckpt}" >&2
        exit 1
    fi

    for eps in "${EPS_LIST[@]}"; do
        local stem="${pgd_dir}/eps${eps}_n${PGD_MAX_SAMPLES}"
        local json="${stem}.json"
        local log="${stem}.log"

        echo
        echo "---- PGD ${label}: eps=${eps}/255, n=${PGD_MAX_SAMPLES}, mod_steps=${PGD_MOD_STEPS} ----"
        echo "     siren : ${ckpt}"
        echo "     class : ${clf_ckpt}"
        echo "     json  : ${json}"
        echo "     log   : ${log}"

        if [[ -f "${json}" ]]; then
            echo "     [skip] found ${json}"
            continue
        fi

        python -u attacks/full_pgd_cifar10_spatial.py \
            --data-path "${DATA_PATH}" \
            --siren-checkpoint "${ckpt}" \
            --classifier-checkpoint "${clf_ckpt}" \
            --epsilon "${eps}" \
            --pgd-steps "${PGD_STEPS}" \
            --mod-steps "${PGD_MOD_STEPS}" \
            --ext-lr "${PGD_LR}" \
            --inner-lr "${PGD_INNER_LR}" \
            --seed "${SEED}" \
            --max-samples "${PGD_MAX_SAMPLES}" \
            --output-json "${json}" \
            --device "${ATTACK_DEVICE}" 2>&1 | tee "${log}"
    done
}

summarize_pgd() {
    local summary_json="${EXP_ROOT}/pgd_inner${PGD_MOD_STEPS}_summary.json"
    local summary_md="${EXP_ROOT}/pgd_inner${PGD_MOD_STEPS}_summary.md"

    python - <<PY_SUMMARY
import glob
import json
import os

exp_root = "${EXP_ROOT}"
runs = [
    ("${VANILLA_LABEL}", "${VANILLA_RUN}/pgd_cifar10_spatial_cnn_inner${PGD_MOD_STEPS}"),
    ("${SOFTLIP_LABEL}", "${SOFTLIP_RUN}/pgd_cifar10_spatial_cnn_inner${PGD_MOD_STEPS}"),
]
records = []
for label, root in runs:
    for path in sorted(glob.glob(os.path.join(root, "eps*_n*.json"))):
        with open(path) as f:
            rec = json.load(f)
        rec["model"] = label
        rec["path"] = path
        rec["epsilon_255"] = int(round(rec["constraint"] * 255))
        records.append(rec)

records.sort(key=lambda r: (r["epsilon_255"], r["model"]))
with open("${summary_json}", "w") as f:
    json.dump(records, f, indent=2)

lines = [
    "# CIFAR-10 Spatial-Functa PGD Summary - inner5",
    "",
    f"Experiment root: `{exp_root}`",
    "",
    f"Attack mod steps: `${PGD_MOD_STEPS}`",
    f"PGD steps: `${PGD_STEPS}`",
    "",
    "| model | eps (/255) | n | clean acc | robust acc | robust | clean | gap |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in records:
    gap = r["clean_acc"] - r["robust_acc"]
    lines.append(
        f"| {r['model']} | {r['epsilon_255']} | {r['n_samples']} | "
        f"{r['clean_acc']:.4f} | {r['robust_acc']:.4f} | "
        f"{r['conditional_robust_acc']:.4f} | {gap:+.4f} |"
    )
with open("${summary_md}", "w") as f:
    f.write("\\n".join(lines) + "\\n")
print("\\n".join(lines))
print()
print("[summary] JSON:", "${summary_json}")
print("[summary] MD  :", "${summary_md}")
PY_SUMMARY
}

if [[ "${RUN_MAKESET}" == "1" ]]; then
    if [[ "${RUN_VANILLA}" == "1" ]]; then
        make_functaset "${VANILLA_LABEL}" "${VANILLA_CKPT}" "${VANILLA_RUN}" "${VANILLA_STEM}" "vanilla"
    fi
    if [[ "${RUN_SOFTLIP}" == "1" ]]; then
        make_functaset "${SOFTLIP_LABEL}" "${SOFTLIP_CKPT}" "${SOFTLIP_RUN}" "${SOFTLIP_STEM}" "soft_lipschitz"
    fi
fi

if [[ "${RUN_CLASSIFIER}" == "1" ]]; then
    if [[ "${RUN_VANILLA}" == "1" ]]; then
        train_cnn_classifier "${VANILLA_LABEL}" "${VANILLA_RUN}" "${VANILLA_STEM}"
    fi
    if [[ "${RUN_SOFTLIP}" == "1" ]]; then
        train_cnn_classifier "${SOFTLIP_LABEL}" "${SOFTLIP_RUN}" "${SOFTLIP_STEM}"
    fi
fi

if [[ "${RUN_PGD}" == "1" ]]; then
    if [[ "${RUN_VANILLA}" == "1" ]]; then
        run_pgd "${VANILLA_LABEL}" "${VANILLA_CKPT}" "${VANILLA_RUN}"
    fi
    if [[ "${RUN_SOFTLIP}" == "1" ]]; then
        run_pgd "${SOFTLIP_LABEL}" "${SOFTLIP_CKPT}" "${SOFTLIP_RUN}"
    fi
    summarize_pgd
fi

cat > "${EXP_ROOT}/run_config.md" <<EOF_CONFIG
# CIFAR-10 Spatial-Functa Inner-5 Run Config

- vanilla checkpoint: \`${VANILLA_CKPT}\`
- softlip checkpoint: \`${SOFTLIP_CKPT}\`
- makeset inner steps: \`${MAKE_ITERS}\`
- makeset inner lr: \`${MAKE_LR}\`
- makeset optimizer: \`${MAKE_INNER_OPTIM}\`
- classifier type: \`cnn\`
- classifier lr: \`${CLF_LR}\`
- classifier width: \`${CLF_CNN_WIDTH}\`
- classifier dropout: \`${CLF_DROPOUT}\`
- classifier normalize phi: \`${CLF_NORMALIZE_PHI}\`
- classifier epochs: \`${CLF_EPOCHS}\`
- PGD mod steps: \`${PGD_MOD_STEPS}\`
- PGD steps: \`${PGD_STEPS}\`
- PGD LR: \`${PGD_LR}\`
- PGD eps list: \`${EPS_STR}\`
- PGD max samples: \`${PGD_MAX_SAMPLES}\`
EOF_CONFIG

echo
echo "Done."
echo "  Experiment root: ${EXP_ROOT}"
echo "  Config         : ${EXP_ROOT}/run_config.md"
echo "  Vanilla root   : ${VANILLA_RUN}"
echo "  Softlip root   : ${SOFTLIP_RUN}"
