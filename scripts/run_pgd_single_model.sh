# #!/bin/bash

# # Full-PGD evaluation for ONE model only (no vanilla baseline).
# #
# # Same attack recipe as scripts/run_pgd_plan.sh, but for one model only.
# # In the common case you only set MODEL_SLUG. You can also override
# # MODEL_CKPT / MODEL_CLASS / PGD_OUT directly when the checkpoint directory
# # and run-artifact directory intentionally differ.
# #
# # Default is the hardcap90 run:
# #   SIREN checkpoint was saved before trainer.py --model-name existed, so it
# #   lives under model_mnist/softlip_L1_lam1e+00_sine_and_readout/.
# #   Classifier + outputs live under runs/softlip_hardcap90_lam1e+00_sine_and_readout/.
# #
# # Generic path convention:
# #   SIREN:     model_mnist/${MODEL_SLUG}/modSiren.pth
# #   classifier runs/${MODEL_SLUG}/mnist_classifier/best_classifier.pth
# #   outputs:   runs/${MODEL_SLUG}/pgd_plan/
# #
# # Stages:
# #   A. Main run   @ eps=MAIN_EPS/255, n=N_MAIN
# #   B. Epsilon sweep @ SWEEP_EPS[] /255, n=N_SWEEP  (MAIN_EPS is not repeated here;
# #      it is covered by stage A — same as run_pgd_plan.sh)
# #   C. Collect JSON summaries -> runs/${MODEL_SLUG}/pgd_plan/summary.{json,md}
# #
# # Wall-clock hint (~27 s/sample on RTX 2080 Ti class GPU):
# #   stage A: N_MAIN * 27s
# #   stage B: len(SWEEP_EPS) * N_SWEEP * 27s

# set -euo pipefail

# # ---- model (edit these, or override from the shell) --------------------------
# MODEL_SLUG="${MODEL_SLUG:-softlip_hardcap90_lam1e+00_sine_and_readout}"

# DEFAULT_MODEL_CKPT="model_mnist/${MODEL_SLUG}/modSiren.pth"
# if [[ "${MODEL_SLUG}" == "softlip_hardcap90_lam1e+00_sine_and_readout" ]]; then
#     # This model was trained before --model-name was wired into trainer.py.
#     DEFAULT_MODEL_CKPT="model_mnist/softlip_L1_lam1e+00_sine_and_readout/modSiren.pth"
# fi

# MODEL_CKPT="${MODEL_CKPT:-${DEFAULT_MODEL_CKPT}}"
# MODEL_CLASS="${MODEL_CLASS:-runs/${MODEL_SLUG}/mnist_classifier/best_classifier.pth}"
# PGD_OUT="${PGD_OUT:-runs/${MODEL_SLUG}/pgd_plan}"

# # ---- knobs (same semantics as run_pgd_plan.sh) -------------------------------
# N_MAIN="${N_MAIN:-500}"
# N_SWEEP="${N_SWEEP:-200}"
# MAIN_EPS="${MAIN_EPS:-16}"
# SWEEP_EPS_STR="${SWEEP_EPS_STR:-8 32 64}"
# read -r -a SWEEP_EPS <<< "${SWEEP_EPS_STR}"

# PGD_STEPS="${PGD_STEPS:-100}"
# PGD_MOD_STEPS="${PGD_MOD_STEPS:-10}"
# PGD_LR="${PGD_LR:-0.01}"
# PGD_INNER_LR="${PGD_INNER_LR:-0.01}"
# SEED="${SEED:-0}"

# CUDA_GPU="${CUDA_GPU:-1}"

# # -----------------------------------------------------------------------------
# source /home/omarg/miniforge3/etc/profile.d/conda.sh
# conda activate pss

# export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

# cd ~/SIREN_Vista || exit 1

# mkdir -p "${PGD_OUT}"

# echo "== PGD single-model plan =="
# echo "  MODEL_SLUG : ${MODEL_SLUG}"
# echo "  siren      : ${MODEL_CKPT}"
# echo "  classifier : ${MODEL_CLASS}"
# echo "  output dir : ${PGD_OUT}"
# echo "  N_MAIN=${N_MAIN}  N_SWEEP=${N_SWEEP}"
# echo "  main eps=${MAIN_EPS}/255   sweep eps=${SWEEP_EPS[*]} /255"
# echo "  PGD_STEPS=${PGD_STEPS}  PGD_MOD_STEPS=${PGD_MOD_STEPS}"
# echo "  CUDA device: ${CUDA_VISIBLE_DEVICES}"
# echo

# python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"
# echo

# for ck in "${MODEL_CKPT}" "${MODEL_CLASS}"; do
#     if [[ ! -f "${ck}" ]]; then
#         echo "ERROR: missing checkpoint ${ck}" >&2
#         exit 1
#     fi
# done

# run_pgd() {
#     local ckpt=$1
#     local cls=$2
#     local eps=$3
#     local n=$4
#     local stem=$5
#     local log="${stem}.log"
#     local json="${stem}.json"

#     echo
#     echo "---- PGD: eps=${eps}/255 n=${n} ----"
#     echo "     siren : ${ckpt}"
#     echo "     class : ${cls}"
#     echo "     log   : ${log}"
#     echo "     json  : ${json}"

#     if [[ -f "${json}" ]]; then
#         echo "     [skip] ${json} already exists; delete it to re-run."
#         return 0
#     fi

#     python -u attacks/full_pgd.py \
#         --dataset mnist \
#         --data-path ../data \
#         --siren-checkpoint "${ckpt}" \
#         --classifier-checkpoint "${cls}" \
#         --epsilon "${eps}" \
#         --pgd-steps "${PGD_STEPS}" \
#         --mod-steps "${PGD_MOD_STEPS}" \
#         --ext-lr "${PGD_LR}" \
#         --inner-lr "${PGD_INNER_LR}" \
#         --cwidth 512 \
#         --cdepth 3 \
#         --mod-dim 512 \
#         --hidden-dim 256 \
#         --depth 10 \
#         --seed "${SEED}" \
#         --max-samples "${n}" \
#         --output-json "${json}" \
#         --device cuda 2>&1 | tee "${log}"
# }

# echo
# echo "#### STAGE A: main run (eps=${MAIN_EPS}/255, n=${N_MAIN}) ####"
# run_pgd "${MODEL_CKPT}" "${MODEL_CLASS}" "${MAIN_EPS}" "${N_MAIN}" \
#         "${PGD_OUT}/eps${MAIN_EPS}_n${N_MAIN}"

# echo
# echo "#### STAGE B: epsilon sweep (n=${N_SWEEP}; eps=${MAIN_EPS} already in stage A) ####"
# for eps in "${SWEEP_EPS[@]}"; do
#     run_pgd "${MODEL_CKPT}" "${MODEL_CLASS}" "${eps}" "${N_SWEEP}" \
#             "${PGD_OUT}/eps${eps}_n${N_SWEEP}"
# done

# echo
# echo "#### STAGE C: summary ####"
# SUMMARY_JSON="${PGD_OUT}/summary.json"
# SUMMARY_MD="${PGD_OUT}/summary.md"

# python - <<PY
# import glob, json, os

# os.chdir(os.path.expanduser("~/SIREN_Vista"))

# root = "${PGD_OUT}"
# slug = "${MODEL_SLUG}"

# runs = []
# for jp in sorted(glob.glob(os.path.join(root, "eps*_n*.json"))):
#     with open(jp) as f:
#         rec = json.load(f)
#     rec["model"] = slug
#     rec["path"] = jp
#     rec["epsilon_255"] = int(round(rec["constraint"] * 255))
#     runs.append(rec)

# runs.sort(key=lambda r: (r["epsilon_255"], r["n_samples"]))

# with open("${SUMMARY_JSON}", "w") as f:
#     json.dump(runs, f, indent=2)

# lines = []
# lines.append("# PGD summary — single model " + slug)
# lines.append("")
# lines.append("Each row is one Full-PGD run (`attacks/full_pgd.py`).")
# lines.append("")
# lines.append("| slug | eps (/255) |   n | clean acc | robust acc | robust \\| clean | gap (clean−robust) |")
# lines.append("|------|-----------:|----:|----------:|-----------:|-----------------:|-------------------:|")
# for r in runs:
#     gap = r["clean_acc"] - r["robust_acc"]
#     lines.append(
#         f"| {slug} | {r['epsilon_255']:>10} | {r['n_samples']:>3} | "
#         f"{r['clean_acc']:>9.4f} | {r['robust_acc']:>10.4f} | "
#         f"{r['conditional_robust_acc']:>16.4f} | {gap:>+18.4f} |"
#     )

# with open("${SUMMARY_MD}", "w") as f:
#     f.write("\n".join(lines) + "\n")

# print("\n".join(lines))
# print()
# print(f"[summary] JSON : ${SUMMARY_JSON}")
# print(f"[summary] MD   : ${SUMMARY_MD}")
# PY

# echo
# echo "Done."
# echo "  Per-run JSONs : ${PGD_OUT}/*.json"
# echo "  Per-run logs  : ${PGD_OUT}/*.log"
# echo "  Summary       : ${SUMMARY_JSON}"
# echo "  Summary MD    : ${SUMMARY_MD}"



#!/bin/bash

# Full-PGD evaluation for ONE model only.
#
# Default model:
#   softlip_first95_rest80_lam1e+00_sine_and_readout
#
# This is the layer-specific cap experiment:
#   sine.0       = 95% of vanilla sigma
#   sine.1-9     = 80% of vanilla sigma
#   readout      = 80% of vanilla sigma
#   modul        = not capped
#
# Default run:
#   eps = 8, 16, 32, 64 / 255
#   n   = 200 for each epsilon
#
# Stage A runs eps=16/255.
# Stage B runs eps=8,32,64/255.
#
# Output:
#   runs/${MODEL_SLUG}/pgd_plan/

set -euo pipefail

# ---- model -------------------------------------------------------------------
MODEL_SLUG="${MODEL_SLUG:-softlip_first95_rest80_lam1e+00_sine_and_readout}"

MODEL_CKPT="${MODEL_CKPT:-model_mnist/${MODEL_SLUG}/modSiren.pth}"
MODEL_CLASS="${MODEL_CLASS:-runs/${MODEL_SLUG}/mnist_classifier/best_classifier.pth}"
PGD_OUT="${PGD_OUT:-runs/${MODEL_SLUG}/pgd_plan}"

# ---- knobs -------------------------------------------------------------------
# Full sweep, all with n=200.
N_MAIN="${N_MAIN:-200}"
N_SWEEP="${N_SWEEP:-200}"

# Main run is eps=16/255.
MAIN_EPS="${MAIN_EPS:-16}"

# Sweep excludes 16 because MAIN_EPS already covers it.
SWEEP_EPS_STR="${SWEEP_EPS_STR:-8 32 64}"
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

CUDA_GPU="${CUDA_GPU:-1}"

# -----------------------------------------------------------------------------
source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

mkdir -p "${PGD_OUT}"

echo "== PGD single-model plan =="
echo "  MODEL_SLUG : ${MODEL_SLUG}"
echo "  siren      : ${MODEL_CKPT}"
echo "  classifier : ${MODEL_CLASS}"
echo "  output dir : ${PGD_OUT}"
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
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"
echo

for ck in "${MODEL_CKPT}" "${MODEL_CLASS}"; do
    if [[ ! -f "${ck}" ]]; then
        echo "ERROR: missing checkpoint ${ck}" >&2
        exit 1
    fi
done

run_pgd() {
    local ckpt=$1
    local cls=$2
    local eps=$3
    local n=$4
    local stem=$5
    local log="${stem}.log"
    local json="${stem}.json"

    echo
    echo "---- PGD: eps=${eps}/255 n=${n} ----"
    echo "     siren : ${ckpt}"
    echo "     class : ${cls}"
    echo "     log   : ${log}"
    echo "     json  : ${json}"

    if [[ -f "${json}" ]]; then
        echo "     [skip] ${json} already exists; delete it to re-run."
        return 0
    fi

    python -u attacks/full_pgd.py \
        --dataset mnist \
        --data-path ../data \
        --siren-checkpoint "${ckpt}" \
        --classifier-checkpoint "${cls}" \
        --epsilon "${eps}" \
        --pgd-steps "${PGD_STEPS}" \
        --mod-steps "${PGD_MOD_STEPS}" \
        --ext-lr "${PGD_LR}" \
        --inner-lr "${PGD_INNER_LR}" \
        --cwidth 512 \
        --cdepth 3 \
        --mod-dim 512 \
        --hidden-dim 256 \
        --depth 10 \
        --seed "${SEED}" \
        --max-samples "${n}" \
        --output-json "${json}" \
        --device cuda 2>&1 | tee "${log}"
}

echo
echo "#### STAGE A: main run (eps=${MAIN_EPS}/255, n=${N_MAIN}) ####"
run_pgd "${MODEL_CKPT}" "${MODEL_CLASS}" "${MAIN_EPS}" "${N_MAIN}" \
        "${PGD_OUT}/eps${MAIN_EPS}_n${N_MAIN}"

echo
echo "#### STAGE B: epsilon sweep (n=${N_SWEEP}; eps=${MAIN_EPS} already in stage A) ####"
for eps in "${SWEEP_EPS[@]}"; do
    run_pgd "${MODEL_CKPT}" "${MODEL_CLASS}" "${eps}" "${N_SWEEP}" \
            "${PGD_OUT}/eps${eps}_n${N_SWEEP}"
done

echo
echo "#### STAGE C: summary ####"
SUMMARY_JSON="${PGD_OUT}/summary.json"
SUMMARY_MD="${PGD_OUT}/summary.md"

python - <<PY
import glob
import json
import os

os.chdir(os.path.expanduser("~/SIREN_Vista"))

root = "${PGD_OUT}"
slug = "${MODEL_SLUG}"

runs = []
for jp in sorted(glob.glob(os.path.join(root, "eps*_n*.json"))):
    with open(jp) as f:
        rec = json.load(f)

    rec["model"] = slug
    rec["path"] = jp
    rec["epsilon_255"] = int(round(rec["constraint"] * 255))
    runs.append(rec)

runs.sort(key=lambda r: (r["epsilon_255"], r["n_samples"]))

with open("${SUMMARY_JSON}", "w") as f:
    json.dump(runs, f, indent=2)

lines = []
lines.append("# PGD summary — single model " + slug)
lines.append("")
lines.append("Each row is one Full-PGD run (`attacks/full_pgd.py`).")
lines.append("")
lines.append("| slug | eps (/255) |   n | clean acc | robust acc | robust \\\\| clean | gap (clean−robust) |")
lines.append("|------|-----------:|----:|----------:|-----------:|-----------------:|-------------------:|")

for r in runs:
    gap = r["clean_acc"] - r["robust_acc"]
    lines.append(
        f"| {slug} | {r['epsilon_255']:>10} | {r['n_samples']:>3} | "
        f"{r['clean_acc']:>9.4f} | {r['robust_acc']:>10.4f} | "
        f"{r['conditional_robust_acc']:>16.4f} | {gap:>+18.4f} |"
    )

with open("${SUMMARY_MD}", "w") as f:
    f.write("\\n".join(lines) + "\\n")

print("\\n".join(lines))
print()
print(f"[summary] JSON : ${SUMMARY_JSON}")
print(f"[summary] MD   : ${SUMMARY_MD}")
PY

echo
echo "Done."
echo "  Per-run JSONs : ${PGD_OUT}/*.json"
echo "  Per-run logs  : ${PGD_OUT}/*.log"
echo "  Summary       : ${SUMMARY_JSON}"
echo "  Summary MD    : ${SUMMARY_MD}"
