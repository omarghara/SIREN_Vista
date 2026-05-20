#!/bin/bash

# PGD robustness comparison plan (softlip vs vanilla, MNIST, Full-PGD).
#
# Runs the four-step plan from context/report.md / attack_currenct_results.md:
#   A. softlip   @ eps=16/255, n=N_MAIN   (main datapoint, tight error bar)
#   B. vanilla   @ eps=16/255, n=N_MAIN   (matched baseline)
#   C. eps sweep @ eps in {8, 32, 64}/255, n=N_SWEEP on both models
#      (eps=16 is reused from stages A/B, so not re-run here)
#   D. JSON + tabular summary of all runs.
#
# All stages call the (modified) attacks/full_pgd.py which emits both the
# unconditional robust accuracy (matches the PSS paper's metric) and the
# conditional "robust | clean correct" accuracy. Summaries are written to
# per-run JSON files and collected into a single ascii table at the end.
#
# NOTE:
#   - If you already have a softlip PGD run in progress, kill it before
#     starting this script (it will try to use the same GPU and might OOM
#     if both run on the same device). e.g.:
#         pkill -9 -f "attacks/full_pgd.py"
#   - Total wall-clock at ~27 s/sample on an RTX 2080 Ti:
#         stage A: N_MAIN * 27s    ~  3h45 at N_MAIN=500
#         stage B: N_MAIN * 27s    ~  3h45 at N_MAIN=500
#         stage C: 2 * 3 * N_SWEEP * 27s  ~  9h00 at N_SWEEP=200
#         total:   ~16h30 at defaults. Lower N_SWEEP to 100 to halve stage C.

set -euo pipefail

# ---- knobs (edit these) -----------------------------------------------------
N_MAIN=500            # sample count for eps=16/255 main runs
N_SWEEP=200           # sample count for epsilon sweep runs
MAIN_EPS=16           # primary reported epsilon (in /255)
SWEEP_EPS=(8 32 64)   # other epsilons for the sweep (16 reused from main)

PGD_STEPS=100         # outer PGD iterations
PGD_MOD_STEPS=10      # unrolled inner modulation steps per PGD iter
PGD_LR=0.01           # PGD optimizer (Adam) learn rate on the perturbation
PGD_INNER_LR=0.01     # inner-loop modulation SGD lr (must match makeset)
SEED=0                # torch/np seed for both runs

CUDA_GPU=1            # which GPU

# softlip-side paths (the run we are actually evaluating)
SOFTLIP_SLUG="softlip_L30_lam1e+00_all_skip0"
SOFTLIP_CKPT="model_mnist/${SOFTLIP_SLUG}/modSiren.pth"
SOFTLIP_CLASS="runs/${SOFTLIP_SLUG}/mnist_classifier/best_classifier.pth"
SOFTLIP_OUT="runs/${SOFTLIP_SLUG}/pgd_plan"

# vanilla-side paths (the baseline we compare against)
VANILLA_CKPT="../Parameter-Space-Attack-Suite/model_mnist/modSiren.pth"
VANILLA_CLASS="../Parameter-Space-Attack-Suite/mnist_classifier/best_classifier.pth"
VANILLA_OUT="runs/vanilla/pgd_plan"
# -----------------------------------------------------------------------------

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

mkdir -p "${SOFTLIP_OUT}" "${VANILLA_OUT}"

echo "== PGD robustness plan =="
echo "  softlip slug   : ${SOFTLIP_SLUG}"
echo "  softlip siren  : ${SOFTLIP_CKPT}"
echo "  softlip class  : ${SOFTLIP_CLASS}"
echo "  vanilla siren  : ${VANILLA_CKPT}"
echo "  vanilla class  : ${VANILLA_CLASS}"
echo "  N_MAIN=${N_MAIN}  N_SWEEP=${N_SWEEP}"
echo "  main eps=${MAIN_EPS}/255   sweep eps=${SWEEP_EPS[*]} /255"
echo "  PGD_STEPS=${PGD_STEPS}  PGD_MOD_STEPS=${PGD_MOD_STEPS}"
echo "  CUDA device    : ${CUDA_VISIBLE_DEVICES}"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"
echo

# Sanity: all four checkpoints must exist before we burn GPU hours.
for ck in "${SOFTLIP_CKPT}" "${SOFTLIP_CLASS}" "${VANILLA_CKPT}" "${VANILLA_CLASS}"; do
    if [[ ! -f "${ck}" ]]; then
        echo "ERROR: missing checkpoint ${ck}" >&2
        exit 1
    fi
done

# run_pgd <siren_ckpt> <classifier_ckpt> <eps_int> <n_samples> <out_stem>
#   - writes <out_stem>.log  (tee'd stdout, includes per-sample tqdm line)
#   - writes <out_stem>.json (final metric summary)
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

    # Skip if already produced (idempotent re-runs).
    if [[ -f "${json}" ]]; then
        echo "     [skip] ${json} already exists; delete it to re-run."
        return 0
    fi

    # -u = unbuffered so tqdm appears live in the tee'd log.
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
echo "#### STAGE A: softlip main run (eps=${MAIN_EPS}/255, n=${N_MAIN}) ####"
run_pgd "${SOFTLIP_CKPT}" "${SOFTLIP_CLASS}" "${MAIN_EPS}" "${N_MAIN}" \
        "${SOFTLIP_OUT}/eps${MAIN_EPS}_n${N_MAIN}"

echo
echo "#### STAGE B: vanilla main run (eps=${MAIN_EPS}/255, n=${N_MAIN}) ####"
run_pgd "${VANILLA_CKPT}" "${VANILLA_CLASS}" "${MAIN_EPS}" "${N_MAIN}" \
        "${VANILLA_OUT}/eps${MAIN_EPS}_n${N_MAIN}"

echo
echo "#### STAGE C: epsilon sweep (n=${N_SWEEP}) ####"
echo "     epsilons : ${SWEEP_EPS[*]} /255  (eps=${MAIN_EPS} reused from A/B)"
for eps in "${SWEEP_EPS[@]}"; do
    run_pgd "${SOFTLIP_CKPT}" "${SOFTLIP_CLASS}" "${eps}" "${N_SWEEP}" \
            "${SOFTLIP_OUT}/eps${eps}_n${N_SWEEP}"
    run_pgd "${VANILLA_CKPT}" "${VANILLA_CLASS}" "${eps}" "${N_SWEEP}" \
            "${VANILLA_OUT}/eps${eps}_n${N_SWEEP}"
done

echo
echo "#### STAGE D: summary ####"
SUMMARY_JSON="runs/pgd_plan_summary.json"
SUMMARY_MD="runs/pgd_plan_summary.md"

# Collect every per-run JSON into a single tabular summary. Using a here-doc
# so we don't need an extra Python file on disk.
python - <<PY
import glob, json, os, sys

os.chdir(os.path.expanduser("~/SIREN_Vista"))

runs = []
for label, root in [
    ("softlip", "${SOFTLIP_OUT}"),
    ("vanilla", "${VANILLA_OUT}"),
]:
    for jp in sorted(glob.glob(os.path.join(root, "eps*_n*.json"))):
        with open(jp) as f:
            rec = json.load(f)
        rec["model"] = label
        rec["path"] = jp
        rec["epsilon_255"] = int(round(rec["constraint"] * 255))
        runs.append(rec)

runs.sort(key=lambda r: (r["epsilon_255"], r["model"]))

# JSON dump
with open("${SUMMARY_JSON}", "w") as f:
    json.dump(runs, f, indent=2)

# Pretty table
lines = []
lines.append("# PGD robustness plan -- summary")
lines.append("")
lines.append("Each row is one (model, epsilon, n_samples) run of attacks/full_pgd.py.")
lines.append("")
lines.append("| model   | eps (/255) |   n | clean acc | robust acc | robust \\| clean | gap (clean-robust) |")
lines.append("|---------|-----------:|----:|----------:|-----------:|-----------------:|-------------------:|")
for r in runs:
    gap = r["clean_acc"] - r["robust_acc"]
    lines.append(
        f"| {r['model']:<7} | {r['epsilon_255']:>10} | {r['n_samples']:>3} | "
        f"{r['clean_acc']:>9.4f} | {r['robust_acc']:>10.4f} | "
        f"{r['conditional_robust_acc']:>16.4f} | {gap:>+18.4f} |"
    )

with open("${SUMMARY_MD}", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print()
print(f"[summary] JSON : ${SUMMARY_JSON}")
print(f"[summary] MD   : ${SUMMARY_MD}")
PY

echo
echo "Done."
echo "  Per-run JSONs : ${SOFTLIP_OUT}/*.json  and  ${VANILLA_OUT}/*.json"
echo "  Per-run logs  : ${SOFTLIP_OUT}/*.log   and  ${VANILLA_OUT}/*.log"
echo "  Summary JSON  : ${SUMMARY_JSON}"
echo "  Summary MD    : ${SUMMARY_MD}"
