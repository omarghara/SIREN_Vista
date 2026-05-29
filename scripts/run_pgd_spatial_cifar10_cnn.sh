#!/bin/bash

# Full-PGD evaluation for CIFAR-10 Spatial-Functa models with CNN classifiers.
#
# Compares two matched spatial SIREN runs:
#   1. spatial paper SIREN e512 + CNN classifier (~70.45% clean top-1)
#   2. spatial paper SIREN soft-Lipschitz e12 + CNN classifier (~66.90% clean top-1)
#
# The attack entry point is attacks/full_pgd_cifar10_spatial.py.
# It loads SpatialModulatedINR checkpoints from metadata and CNN classifier
# checkpoints from train_classifier.py metadata, including stored phi
# normalization stats.

set -euo pipefail

# ---- model slugs -------------------------------------------------------------

VANILLA_SLUG="${VANILLA_SLUG:-functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000}"
SOFTLIP_SLUG="${SOFTLIP_SLUG:-functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cap90_lam1e-02_sine_and_readout_train50000_test10000}"

VANILLA_CKPT="${VANILLA_CKPT:-model_cifar10/${VANILLA_SLUG}/modSiren.pth}"
VANILLA_CLASS="${VANILLA_CLASS:-runs/${VANILLA_SLUG}/cifar10_cnn_classifier/best_classifier.pth}"
VANILLA_OUT="${VANILLA_OUT:-runs/${VANILLA_SLUG}/pgd_cifar10_spatial_cnn}"

SOFTLIP_CKPT="${SOFTLIP_CKPT:-model_cifar10/${SOFTLIP_SLUG}/modSiren.pth}"
SOFTLIP_CLASS="${SOFTLIP_CLASS:-runs/${SOFTLIP_SLUG}/cifar10_cnn_classifier/best_classifier.pth}"
SOFTLIP_OUT="${SOFTLIP_OUT:-runs/${SOFTLIP_SLUG}/pgd_cifar10_spatial_cnn}"

# ---- knobs -------------------------------------------------------------------
# CIFAR-10 spatial Full-PGD is expensive. Start small, then raise these.
N_MAIN="${N_MAIN:-100}"
N_SWEEP="${N_SWEEP:-50}"

MAIN_EPS="${MAIN_EPS:-16}"
SWEEP_EPS_STR="${SWEEP_EPS_STR-8 32 64}"
if [[ -n "${SWEEP_EPS_STR}" ]]; then
    read -r -a SWEEP_EPS <<< "${SWEEP_EPS_STR}"
else
    SWEEP_EPS=()
fi

PGD_STEPS="${PGD_STEPS:-100}"
PGD_MOD_STEPS="${PGD_MOD_STEPS:-10}"
PGD_LR="${PGD_LR:-0.01}"
PGD_INNER_LR="${PGD_INNER_LR:-0.01}"
SEED="${SEED:-0}"

CUDA_GPU="${CUDA_GPU:-0}"
ATTACK_DEVICE="${ATTACK_DEVICE:-cuda}"

# Set RUN_VANILLA=0 or RUN_SOFTLIP=0 to attack only one model.
RUN_VANILLA="${RUN_VANILLA:-1}"
RUN_SOFTLIP="${RUN_SOFTLIP:-1}"

# -----------------------------------------------------------------------------
source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

mkdir -p "${VANILLA_OUT}" "${SOFTLIP_OUT}"

echo "== CIFAR-10 Spatial-Functa CNN Full-PGD plan =="
echo "  vanilla slug : ${VANILLA_SLUG}"
echo "  vanilla ckpt : ${VANILLA_CKPT}"
echo "  vanilla cls  : ${VANILLA_CLASS}"
echo "  softlip slug : ${SOFTLIP_SLUG}"
echo "  softlip ckpt : ${SOFTLIP_CKPT}"
echo "  softlip cls  : ${SOFTLIP_CLASS}"
echo "  N_MAIN=${N_MAIN}  N_SWEEP=${N_SWEEP}"
echo "  main eps=${MAIN_EPS}/255"
if [[ "${#SWEEP_EPS[@]}" -gt 0 ]]; then
    echo "  sweep eps=${SWEEP_EPS[*]} /255"
else
    echo "  sweep eps=(none)"
fi
echo "  PGD_STEPS=${PGD_STEPS}  PGD_MOD_STEPS=${PGD_MOD_STEPS}"
echo "  PGD_LR=${PGD_LR}  PGD_INNER_LR=${PGD_INNER_LR}"
echo "  SEED=${SEED}"
echo "  CUDA device: ${CUDA_VISIBLE_DEVICES}"
echo "  attack device: ${ATTACK_DEVICE}"
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
echo

for ck in \
    $( [[ "${RUN_VANILLA}" == "1" ]] && echo "${VANILLA_CKPT} ${VANILLA_CLASS}" ) \
    $( [[ "${RUN_SOFTLIP}" == "1" ]] && echo "${SOFTLIP_CKPT} ${SOFTLIP_CLASS}" ); do
    if [[ -n "${ck}" && ! -f "${ck}" ]]; then
        echo "ERROR: missing checkpoint ${ck}" >&2
        exit 1
    fi
done

run_pgd() {
    local label=$1
    local ckpt=$2
    local cls=$3
    local eps=$4
    local n=$5
    local stem=$6
    local log="${stem}.log"
    local json="${stem}.json"

    echo
    echo "---- PGD ${label}: eps=${eps}/255 n=${n} ----"
    echo "     siren : ${ckpt}"
    echo "     class : ${cls}"
    echo "     log   : ${log}"
    echo "     json  : ${json}"

    if [[ -f "${json}" ]]; then
        echo "     [skip] ${json} already exists; delete it to re-run."
        return 0
    fi

    python -u attacks/full_pgd_cifar10_spatial.py         --data-path ../data         --siren-checkpoint "${ckpt}"         --classifier-checkpoint "${cls}"         --epsilon "${eps}"         --pgd-steps "${PGD_STEPS}"         --mod-steps "${PGD_MOD_STEPS}"         --ext-lr "${PGD_LR}"         --inner-lr "${PGD_INNER_LR}"         --seed "${SEED}"         --max-samples "${n}"         --output-json "${json}"         --device "${ATTACK_DEVICE}" 2>&1 | tee "${log}"
}

echo
echo "#### STAGE A: main eps=${MAIN_EPS}/255, n=${N_MAIN} ####"
if [[ "${RUN_VANILLA}" == "1" ]]; then
    run_pgd "vanilla_spatial_siren_cnn" "${VANILLA_CKPT}" "${VANILLA_CLASS}" "${MAIN_EPS}" "${N_MAIN}"         "${VANILLA_OUT}/eps${MAIN_EPS}_n${N_MAIN}"
fi
if [[ "${RUN_SOFTLIP}" == "1" ]]; then
    run_pgd "softlip_spatial_siren_cnn" "${SOFTLIP_CKPT}" "${SOFTLIP_CLASS}" "${MAIN_EPS}" "${N_MAIN}"         "${SOFTLIP_OUT}/eps${MAIN_EPS}_n${N_MAIN}"
fi

echo
echo "#### STAGE B: epsilon sweep (n=${N_SWEEP}; eps=${MAIN_EPS} already in stage A) ####"
for eps in "${SWEEP_EPS[@]}"; do
    if [[ "${RUN_VANILLA}" == "1" ]]; then
        run_pgd "vanilla_spatial_siren_cnn" "${VANILLA_CKPT}" "${VANILLA_CLASS}" "${eps}" "${N_SWEEP}"             "${VANILLA_OUT}/eps${eps}_n${N_SWEEP}"
    fi
    if [[ "${RUN_SOFTLIP}" == "1" ]]; then
        run_pgd "softlip_spatial_siren_cnn" "${SOFTLIP_CKPT}" "${SOFTLIP_CLASS}" "${eps}" "${N_SWEEP}"             "${SOFTLIP_OUT}/eps${eps}_n${N_SWEEP}"
    fi
done

echo
echo "#### STAGE C: summary ####"
SUMMARY_JSON="runs/pgd_cifar10_spatial_cnn_summary.json"
SUMMARY_MD="runs/pgd_cifar10_spatial_cnn_summary.md"

python - <<PY_SUMMARY
import glob
import json
import os

os.chdir(os.path.expanduser("~/SIREN_Vista"))

roots = []
if ${RUN_VANILLA}:
    roots.append(("vanilla_spatial_siren_cnn", "${VANILLA_OUT}"))
if ${RUN_SOFTLIP}:
    roots.append(("softlip_spatial_siren_cnn", "${SOFTLIP_OUT}"))

runs = []
for label, root in roots:
    for jp in sorted(glob.glob(os.path.join(root, "eps*_n*.json"))):
        with open(jp) as f:
            rec = json.load(f)
        rec["model"] = label
        rec["path"] = jp
        rec["epsilon_255"] = int(round(rec["constraint"] * 255))
        runs.append(rec)

runs.sort(key=lambda r: (r["epsilon_255"], r["model"], r["n_samples"]))

with open("${SUMMARY_JSON}", "w") as f:
    json.dump(runs, f, indent=2)

lines = []
lines.append("# CIFAR-10 Spatial-Functa CNN PGD Summary")
lines.append("")
lines.append("Each row is one Full-PGD run of attacks/full_pgd_cifar10_spatial.py.")
lines.append("")
lines.append("| model | eps (/255) | n | clean acc | robust acc | robust \\| clean | gap (clean-robust) |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for r in runs:
    gap = r["clean_acc"] - r["robust_acc"]
    lines.append(
        f"| {r['model']} | {r['epsilon_255']} | {r['n_samples']} | "
        f"{r['clean_acc']:.4f} | {r['robust_acc']:.4f} | "
        f"{r['conditional_robust_acc']:.4f} | {gap:+.4f} |"
    )

with open("${SUMMARY_MD}", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print()
print(f"[summary] JSON : ${SUMMARY_JSON}")
print(f"[summary] MD   : ${SUMMARY_MD}")
PY_SUMMARY

echo
echo "Done."
echo "  Vanilla JSONs : ${VANILLA_OUT}/*.json"
echo "  Softlip JSONs : ${SOFTLIP_OUT}/*.json"
echo "  Summary JSON  : ${SUMMARY_JSON}"
echo "  Summary MD    : ${SUMMARY_MD}"
