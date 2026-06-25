#!/usr/bin/env python
"""Adversarial amplification across pipeline layers.

For each model (INR backbone + spatial-phi CNN classifier) this:
  1. builds a PGD adversarial image (loss = -CE on the classifier),
  2. fits phi_clean and phi_adv,
  3. captures per-layer activations of the INR (phi, sine_0..L, readout) and
     optionally the classifier, for clean vs adversarial,
  4. measures the per-layer activation difference ||a_adv - a_clean||_2 and the
     layer amplification ratio R_l = ||Δa_l|| / ||Δa_{l-1}|| (first layer uses
     the input perturbation L2 as denominator),
  5. averages over N samples and plots all models overlaid.

This mirrors notebooks/cifar10_latest_robustness_layer_analysis.ipynb so the
projected (hard SVD cap) runs can be compared against vanilla.

Example:
    python scripts/amplification_analysis.py --n-samples 16 --epsilon 4 \
      --model "vanilla:SIREN.pth:CLF.pth" \
      --model "readout:SIREN.pth:CLF.pth" \
      --out-dir runs/diagnostics/svdproj
"""

import argparse
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataloader import get_cifar10_loader
from utils import set_random_seeds
from attacks.full_pgd_cifar10_spatial import (
    FullPGDCIFAR10Spatial,
    check_model_classifier_compat,
    load_classifier_checkpoint,
    load_inr_checkpoint,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tensor_l2(x):
    return float(x.detach().float().reshape(-1).norm().cpu().item())


def generate_adversarial_pair(attack_model, x, y, epsilon, pgd_steps, pgd_lr, criterion):
    x = x.to(DEVICE).float()
    y = y.to(DEVICE).long()

    clean_logits, clean_phi, clean_mse = attack_model(x, clean=True, return_mse=True)
    clean_pred = int(clean_logits.argmax(1).item())

    pert = torch.zeros_like(x[0], device=DEVICE, requires_grad=True)
    optimizer = optim.Adam([pert], lr=pgd_lr)
    for _ in range(pgd_steps):
        optimizer.zero_grad()
        proj_pert = torch.clamp(pert, -epsilon, epsilon)
        attacked_x = torch.clamp(x + proj_pert.unsqueeze(0), 0.0, 1.0)
        logits, _ = attack_model(attacked_x, clean_phi.detach(), clean=False)
        loss = -criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        proj_pert = torch.clamp(pert.detach(), -epsilon, epsilon)
        final_x = torch.clamp(x + proj_pert.unsqueeze(0), 0.0, 1.0)
        final_pert = final_x - x
    adv_logits, adv_phi, adv_mse = attack_model(
        final_x, clean_phi.detach(), clean=True, return_mse=True)
    adv_pred = int(adv_logits.argmax(1).item())

    return {
        "y": int(y.item()),
        "clean_pred": clean_pred,
        "adv_pred": adv_pred,
        "clean_phi": clean_phi.detach(),
        "adv_phi": adv_phi.detach(),
        "clean_mse": float(clean_mse) if clean_mse is not None else None,
        "adv_mse": float(adv_mse) if adv_mse is not None else None,
        "delta_norm": tensor_l2(final_pert),
    }


def capture_inr_activations(inr, phi_batch):
    """Per-layer INR activations for one fitted phi (phi, sine_0..L, readout)."""
    phi = phi_batch[0].to(DEVICE).float()
    activations = OrderedDict()
    activations["phi"] = phi.detach().cpu()
    handles = []

    def hook_for(name):
        def hook(module, inputs, output):
            activations[name] = output.detach().cpu()
        return hook

    for i, layer in enumerate(inr.siren.net):
        handles.append(layer.register_forward_hook(hook_for(f"sine_{i}")))
    handles.append(inr.siren.hidden2rgb.register_forward_hook(hook_for("readout_rgb")))
    with torch.no_grad():
        _ = inr(phi)
    for handle in handles:
        handle.remove()
    return activations


def metric_rows(model_label, sample_id, clean_acts, adv_acts, delta_norm):
    """Per-layer diff_norm, diff/||delta||, and amplification ratio R_l."""
    rows = []
    previous_diff = None
    for layer_index, (layer, clean_act) in enumerate(clean_acts.items()):
        if layer not in adv_acts:
            continue
        adv_act = adv_acts[layer]
        diff = tensor_l2(adv_act - clean_act)
        clean_norm = tensor_l2(clean_act)
        if previous_diff is None:
            ratio = diff / max(delta_norm, 1e-12)
        else:
            ratio = diff / max(previous_diff, 1e-12)
        rows.append({
            "model": model_label,
            "sample_id": sample_id,
            "layer_index": layer_index,
            "layer": layer,
            "diff_norm": diff,
            "diff_over_delta": diff / max(delta_norm, 1e-12),
            "amplification_ratio": ratio,
            "clean_norm": clean_norm,
        })
        previous_diff = diff
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True,
                   help="label:siren_ckpt:classifier_ckpt (repeatable).")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--epsilon", type=int, default=4, help="L_inf budget in /255.")
    p.add_argument("--pgd-steps", type=int, default=100)
    p.add_argument("--pgd-lr", type=float, default=0.01)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--inner-lr", type=float, default=0.01)
    p.add_argument("--data-path", type=str, default="../data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="runs/diagnostics/svdproj")
    args = p.parse_args()

    set_random_seeds(args.seed, DEVICE)
    os.makedirs(args.out_dir, exist_ok=True)
    eps = args.epsilon / 255.0
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    models = [m.split(":", 2) for m in args.model]
    tag = f"eps{args.epsilon}_n{args.n_samples}_pgd{args.pgd_steps}_inner{args.inner_steps}"

    loader = get_cifar10_loader(args.data_path, train=False, batch_size=1,
                                num_workers=0, pin_memory=False)

    all_rows = []
    attack_summary = []
    for label, siren_ckpt, clf_ckpt in models:
        print(f"[load] {label}")
        inr, _ = load_inr_checkpoint(siren_ckpt, DEVICE)
        classifier, _ = load_classifier_checkpoint(clf_ckpt, DEVICE)
        check_model_classifier_compat(inr, classifier)
        attack_model = FullPGDCIFAR10Spatial(
            inr, classifier, inner_steps=args.inner_steps, inner_lr=args.inner_lr,
            clean_grad_clip=0.0, device=DEVICE,
        ).to(DEVICE)

        clean_ok = adv_ok = 0
        for sid, (x, y) in enumerate(loader):
            if sid >= args.n_samples:
                break
            pair = generate_adversarial_pair(attack_model, x, y, eps,
                                             args.pgd_steps, args.pgd_lr, criterion)
            clean_ok += int(pair["clean_pred"] == pair["y"])
            adv_ok += int(pair["adv_pred"] == pair["y"])
            clean_acts = capture_inr_activations(inr, pair["clean_phi"])
            adv_acts = capture_inr_activations(inr, pair["adv_phi"])
            all_rows.extend(metric_rows(label, sid, clean_acts, adv_acts, pair["delta_norm"]))
        attack_summary.append({"model": label, "n": args.n_samples,
                               "clean_acc": clean_ok / args.n_samples,
                               "robust_acc": adv_ok / args.n_samples})
        print(f"  clean_acc={clean_ok/args.n_samples:.3f} robust_acc={adv_ok/args.n_samples:.3f}")
        del attack_model, classifier, inr
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    avg = (df.groupby(["model", "layer_index", "layer"], as_index=False)
             .agg(diff_norm=("diff_norm", "mean"),
                  diff_over_delta=("diff_over_delta", "mean"),
                  amplification_ratio=("amplification_ratio", "mean"),
                  clean_norm=("clean_norm", "mean")))
    csv_path = os.path.join(args.out_dir, f"amplification_{tag}.csv")
    avg.to_csv(csv_path, index=False)
    pd.DataFrame(attack_summary).to_csv(
        os.path.join(args.out_dir, f"amplification_attack_summary_{tag}.csv"), index=False)

    def plot(metric, ylabel, fname, logy=True):
        fig, ax = plt.subplots(figsize=(11, 6))
        order = None
        for label in [m[0] for m in models]:
            g = avg[avg["model"] == label].sort_values("layer_index")
            if g.empty:
                continue
            ax.plot(g["layer_index"], g[metric], marker="o", linewidth=1.9, label=label)
            order = g
        if order is not None:
            ax.set_xticks(order["layer_index"])
            ax.set_xticklabels(order["layer"], rotation=35, ha="right")
        ax.set_xlabel("pipeline layer")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Adversarial amplification across INR layers ({tag})")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print("saved:", out)

    plot("diff_norm", r"mean $||a_{adv}-a_{clean}||_2$", f"inr_diff_norm_{tag}.png")
    plot("diff_over_delta", r"mean $||a_{adv}-a_{clean}||_2 / ||\delta||_2$",
         f"inr_diff_over_delta_{tag}.png")
    plot("amplification_ratio", "mean layer amplification ratio $R_l$",
         f"inr_amplification_ratio_{tag}.png")

    print("\n=== attack summary ===")
    print(pd.DataFrame(attack_summary).to_string(index=False))
    print("\n=== per-layer mean diff_norm (pivot) ===")
    print(avg.pivot_table(index="model", columns="layer", values="diff_norm")
             .reindex(columns=[r["layer"] for _, r in
                               avg.sort_values("layer_index").drop_duplicates("layer").iterrows()])
             .to_string())
    print("\ncsv:", csv_path)


if __name__ == "__main__":
    main()
