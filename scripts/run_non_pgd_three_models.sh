#!/bin/bash

# Non-PGD attack evaluation for three MNIST models.
#
# Models:
#   vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256
#   softlip_hardcap90_lam1e+00_sine_and_readout
#   softlip_first95_rest80_lam1e+00_sine_and_readout
#
# Default attack set intentionally excludes attacks/full_pgd.py because that
# sweep is already tracked under pgd_plan/. AutoAttack is also excluded here
# because it has a different batch/all-data driver in attacks/run_autoattack.py.
#
# Outputs:
#   runs/non_pgd_three_models/<model>/<attack>/eps${eps}_n${N}.{log,json}
#   runs/non_pgd_three_models/summary.{json,md}

set -euo pipefail

# ---- knobs -------------------------------------------------------------------
N_SAMPLES="${N_SAMPLES:-100}"
EPS_STR="${EPS_STR:-16}"
ATTACKS_STR="${ATTACKS_STR:-implicit_no_lbfgs implicit icop bottom tmo bpda}"

PGD_STEPS="${PGD_STEPS:-100}"
PGD_MOD_STEPS="${PGD_MOD_STEPS:-10}"
PGD_LR="${PGD_LR:-0.01}"
PGD_INNER_LR="${PGD_INNER_LR:-0.01}"
TMO_STEPS="${TMO_STEPS:-5}"
INTERLEAVE_STEPS="${INTERLEAVE_STEPS:-5}"
PROJ_LR="${PROJ_LR:-0.0005}"
MAX_PROJ_ITERS="${MAX_PROJ_ITERS:-10}"
BPDA_MODE="${BPDA_MODE:-identity}"
BPDA_EOT_SAMPLES="${BPDA_EOT_SAMPLES:-1}"
BPDA_RESTARTS="${BPDA_RESTARTS:-1}"
SEED="${SEED:-0}"

CUDA_GPU="${CUDA_GPU:-1}"
OUT_ROOT="${OUT_ROOT:-runs/non_pgd_three_models}"

read -r -a EPS_LIST <<< "${EPS_STR}"
read -r -a ATTACKS <<< "${ATTACKS_STR}"

# ---- environment --------------------------------------------------------------
source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1
mkdir -p "${OUT_ROOT}"

echo "== non-PGD three-model attack plan =="
echo "  models       : vanilla_e40, softlip_hardcap90, softlip_first95_rest80"
echo "  attacks      : ${ATTACKS[*]}"
echo "  eps          : ${EPS_LIST[*]} /255"
echo "  n samples    : ${N_SAMPLES}"
echo "  output root  : ${OUT_ROOT}"
echo "  CUDA device  : ${CUDA_VISIBLE_DEVICES}"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"
echo

# ---- model registry -----------------------------------------------------------
declare -A MODEL_CKPTS
declare -A MODEL_CLASSES

MODEL_SLUGS=(
    "vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256"
    "softlip_hardcap90_lam1e+00_sine_and_readout"
    "softlip_first95_rest80_lam1e+00_sine_and_readout"
)

MODEL_CKPTS["vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256"]="model_mnist/vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256/modSiren.pth"
MODEL_CKPTS["softlip_hardcap90_lam1e+00_sine_and_readout"]="model_mnist/softlip_L1_lam1e+00_sine_and_readout/modSiren.pth"
MODEL_CKPTS["softlip_first95_rest80_lam1e+00_sine_and_readout"]="model_mnist/softlip_first95_rest80_lam1e+00_sine_and_readout/modSiren.pth"

MODEL_CLASSES["vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256"]="runs/vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256/mnist_classifier/best_classifier.pth"
MODEL_CLASSES["softlip_hardcap90_lam1e+00_sine_and_readout"]="runs/softlip_hardcap90_lam1e+00_sine_and_readout/mnist_classifier/best_classifier.pth"
MODEL_CLASSES["softlip_first95_rest80_lam1e+00_sine_and_readout"]="runs/softlip_first95_rest80_lam1e+00_sine_and_readout/mnist_classifier/best_classifier.pth"

for slug in "${MODEL_SLUGS[@]}"; do
    for ck in "${MODEL_CKPTS[$slug]}" "${MODEL_CLASSES[$slug]}"; do
        if [[ ! -f "${ck}" ]]; then
            echo "ERROR: missing checkpoint ${ck}" >&2
            exit 1
        fi
    done
done

run_attack() {
    local slug=$1
    local attack=$2
    local eps=$3

    local model_dir="${OUT_ROOT}/${slug}/${attack}"
    local stem="${model_dir}/eps${eps}_n${N_SAMPLES}"
    local log="${stem}.log"
    local json="${stem}.json"

    mkdir -p "${model_dir}"

    echo
    echo "---- ${attack}: model=${slug} eps=${eps}/255 n=${N_SAMPLES} ----"
    echo "     siren : ${MODEL_CKPTS[$slug]}"
    echo "     class : ${MODEL_CLASSES[$slug]}"
    echo "     log   : ${log}"
    echo "     json  : ${json}"

    if [[ -f "${json}" ]]; then
        echo "     [skip] ${json} already exists; delete it to re-run."
        return 0
    fi

    python -u - <<PY 2>&1 | tee "${log}"
import argparse
import json
import os
import re
import sys

import torch
from torch.utils.data import DataLoader, Subset

repo = os.path.expanduser("~/SIREN_Vista")
os.chdir(repo)
sys.path.insert(0, repo)
sys.path.insert(0, os.path.join(repo, "attacks"))

from dataloader import get_mnist_loader
from SIREN import ModulatedSIREN
from train_classifier import Classifier
from utils import set_random_seeds

import implicit as implicit_attack
import implicit_no_lbfgs as implicit_no_lbfgs_attack
import icop as icop_attack
import bottom as bottom_attack
import tmo as tmo_attack
import bpda as bpda_attack

slug = "${slug}"
attack = "${attack}"
siren_checkpoint = "${MODEL_CKPTS[$slug]}"
classifier_checkpoint = "${MODEL_CLASSES[$slug]}"
eps_int = int("${eps}")
n_samples = int("${N_SAMPLES}")
seed = int("${SEED}")
device = "cuda"

set_random_seeds(seed, device)

base_loader = get_mnist_loader("../data", train=False, batch_size=1, fashion=False)
subset = Subset(base_loader.dataset, list(range(min(n_samples, len(base_loader.dataset)))))
loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

mod_siren = ModulatedSIREN(height=28, width=28, hidden_features=256, num_layers=10, modul_features=512)
siren_ckpt = torch.load(siren_checkpoint, map_location=device)
mod_siren.load_state_dict(siren_ckpt["state_dict"])

classifier = Classifier(width=512, depth=3, in_features=512, num_classes=10).to(device)
classifier_ckpt = torch.load(classifier_checkpoint, map_location=device)
classifier.load_state_dict(classifier_ckpt["state_dict"])
classifier.eval()

constraint = eps_int / 255.0
criterion = torch.nn.CrossEntropyLoss().to(device)

if attack == "implicit":
    attack_model = implicit_attack.Implicit(
        mod_siren, classifier, inner_steps=int("${PGD_MOD_STEPS}"),
        inner_lr=float("${PGD_INNER_LR}"), device=device
    )
    attack_model.to(device)
    implicit_attack.run_attack(
        attack_model, loader, criterion, constraint,
        int("${PGD_STEPS}"), float("${PGD_LR}"), False, device
    )
elif attack == "implicit_no_lbfgs":
    attack_model = implicit_no_lbfgs_attack.Implicit(
        mod_siren, classifier, inner_steps=int("${PGD_MOD_STEPS}"),
        inner_lr=float("${PGD_INNER_LR}"), voxels=False, device=device
    )
    attack_model.to(device)
    implicit_no_lbfgs_attack.run_attack(
        attack_model, loader, criterion, constraint,
        int("${PGD_STEPS}"), float("${PGD_LR}"), False, device
    )
elif attack == "icop":
    attack_model = icop_attack.ICOP(
        mod_siren, classifier, inner_steps=int("${PGD_MOD_STEPS}"),
        inner_lr=float("${PGD_INNER_LR}"), voxels=False, device=device
    )
    attack_model.to(device)
    icop_attack.run_attack(
        attack_model, loader, criterion, constraint, 512,
        int("${PGD_STEPS}"), float("${PGD_LR}"),
        float("${PROJ_LR}"), int(float("${MAX_PROJ_ITERS}")), False, device
    )
elif attack == "bottom":
    if int("${PGD_MOD_STEPS}") % int("${INTERLEAVE_STEPS}") != 0:
        raise ValueError("PGD_MOD_STEPS must be divisible by INTERLEAVE_STEPS for BOTTOM")
    attack_model = bottom_attack.BOTTOM(
        mod_siren, classifier, bottom_steps=int("${INTERLEAVE_STEPS}"),
        inner_steps=int("${PGD_MOD_STEPS}"), inner_lr=float("${PGD_INNER_LR}"),
        voxels=False, device=device
    )
    attack_model.to(device)
    bottom_attack.run_attack(
        attack_model, loader, criterion, constraint,
        int("${PGD_STEPS}"), float("${PGD_LR}"), False, device
    )
elif attack == "tmo":
    attack_model = tmo_attack.TMO(
        mod_siren, classifier, tmo_steps=int("${TMO_STEPS}"),
        inner_steps=int("${PGD_MOD_STEPS}"), inner_lr=float("${PGD_INNER_LR}"),
        voxels=False, device=device
    )
    attack_model.to(device)
    tmo_attack.run_attack(
        attack_model, loader, criterion, constraint,
        int("${PGD_STEPS}"), float("${PGD_LR}"), False, device
    )
elif attack == "bpda":
    attack_model = bpda_attack.FullPGD(
        mod_siren, classifier, inner_steps=int("${PGD_MOD_STEPS}"),
        inner_lr=float("${PGD_INNER_LR}"), device=device
    )
    attack_model.to(device)
    attack_model.eval()

    clean_correct = 0
    seen = 0
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            logits, _, _ = attack_model(x.unsqueeze(1), clean=True, return_mse=True)
            clean_correct += int((logits.argmax(1) == labels).item())
            seen += 1
    clean_acc = clean_correct / max(1, seen)

    robust_acc = bpda_attack.run_bpda_attack(
        attack_model, loader, constraint, int("${PGD_STEPS}"), float("${PGD_LR}"),
        device, "${BPDA_MODE}", int("${BPDA_EOT_SAMPLES}"),
        int("${BPDA_RESTARTS}"), 0, n_samples
    )
    print(f"Constraint {constraint}: Final clean acc {clean_acc}; Final attacked acc {robust_acc}.")
else:
    raise ValueError(f"Unsupported attack: {attack}")
PY
}

for slug in "${MODEL_SLUGS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
        for eps in "${EPS_LIST[@]}"; do
            run_attack "${slug}" "${attack}" "${eps}"
        done
    done
done

echo
echo "#### summary ####"
SUMMARY_JSON="${OUT_ROOT}/summary.json"
SUMMARY_MD="${OUT_ROOT}/summary.md"

python - <<PY
import glob
import json
import os
import re

root = "${OUT_ROOT}"

rows = []
for log_path in sorted(glob.glob(os.path.join(root, "*", "*", "eps*_n*.log"))):
    parts = log_path.split(os.sep)
    slug = parts[-3]
    attack = parts[-2]
    base = os.path.basename(log_path)
    m_name = re.match(r"eps(?P<eps>\\d+)_n(?P<n>\\d+)\\.log$", base)
    if not m_name:
        continue

    text = open(log_path, errors="replace").read()
    matches = re.findall(
        r"Final clean acc\\s+([0-9.eE+-]+);\\s+Final attacked acc\\s+([0-9.eE+-]+)",
        text,
    )
    status = "ok" if matches else "missing_final_metrics"
    clean_acc = robust_acc = cond_robust = gap = None
    if matches:
        clean_acc = float(matches[-1][0])
        robust_acc = float(matches[-1][1])
        cond_robust = robust_acc / clean_acc if clean_acc > 0 else 0.0
        gap = clean_acc - robust_acc

    row = {
        "model": slug,
        "attack": attack,
        "epsilon_255": int(m_name.group("eps")),
        "n_samples": int(m_name.group("n")),
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "conditional_robust_acc": cond_robust,
        "gap": gap,
        "status": status,
        "log": log_path,
    }
    rows.append(row)

    json_path = log_path[:-4] + ".json"
    with open(json_path, "w") as f:
        json.dump(row, f, indent=2)

rows.sort(key=lambda r: (r["model"], r["attack"], r["epsilon_255"], r["n_samples"]))

with open("${SUMMARY_JSON}", "w") as f:
    json.dump(rows, f, indent=2)

lines = []
lines.append("# Non-PGD attack summary — three models")
lines.append("")
lines.append("Each row is one non-PGD attack run. `robust | clean` is computed as robust_acc / clean_acc from the attack log.")
lines.append("")
lines.append("| model | attack | eps (/255) |   n | clean acc | robust acc | robust \\\\| clean | gap (clean−robust) | status |")
lines.append("|-------|--------|-----------:|----:|----------:|-----------:|-----------------:|-------------------:|--------|")
for r in rows:
    def fmt(x, width):
        return " " * (width - 3) + "n/a" if x is None else f"{x:>{width}.4f}"

    gap_text = " " * 15 + "n/a" if r["gap"] is None else f"{r['gap']:>+18.4f}"
    lines.append(
        f"| {r['model']} | {r['attack']} | {r['epsilon_255']:>10} | {r['n_samples']:>3} | "
        f"{fmt(r['clean_acc'], 9)} | {fmt(r['robust_acc'], 10)} | "
        f"{fmt(r['conditional_robust_acc'], 16)} | "
        f"{gap_text} | "
        f"{r['status']} |"
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
echo "  Per-run JSONs : ${OUT_ROOT}/*/*/*.json"
echo "  Per-run logs  : ${OUT_ROOT}/*/*/*.log"
echo "  Summary       : ${SUMMARY_JSON}"
echo "  Summary MD    : ${SUMMARY_MD}"
