#!/usr/bin/env python
"""Reconstruct one CIFAR-10 image with several trained SIREN backbones and
report PSNR at chosen inner-loop iteration counts (default 3 and 5).

Produces a side-by-side figure (original + reconstruction at each iter count
per model) and prints a PSNR table. The inner-loop fit mirrors makeset.py:
zero-init phi, SGD, lr 0.01.

Example:
    python scripts/reconstruct_compare.py \
        --sample-index 7 --iters 3,5 \
        --model vanilla:model_cifar10/<vanilla>/modSiren.pth \
        --model readout:model_cifar10/<readout>/modSiren.pth \
        --model whole_siren:model_cifar10/<whole>/modSiren.pth \
        --output runs/diagnostics/recon_compare.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataloader import get_cifar10_loader
from SIREN import SpatialModulatedINR
from utils import set_random_seeds


def build_inr(ckpt, device):
    ma = ckpt.get("model_args", {}) or {}
    model = SpatialModulatedINR(
        height=ma.get("height", 32), width=ma.get("width", 32),
        hidden_features=ma["hidden_dim"], num_layers=ma["depth"],
        latent_spatial_dim=ma.get("latent_spatial_dim", 8),
        latent_dim=ma.get("latent_dim", 16),
        base_inr_type=ma.get("inr_type", "siren"),
        spatial_interp=ma.get("spatial_interp", "nearest"),
        use_local_coords=ma.get("use_local_coords", True),
        modulation_type=ma.get("modulation_type", "shift"),
        freq=ma.get("freq", 10.0), device=device,
        out_features=ma.get("out_features", 3),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def fit_and_snapshot(model, target_hwc, iters, inner_lr, device):
    """Fit phi to one image; return {it: (recon_hwc, psnr)} for it in iters."""
    H, W, C = target_hwc.shape
    target = target_hwc.reshape(-1, C).to(device)              # (N, C)
    phi = model.init_phi(device=device).float().detach().requires_grad_(True)
    opt = optim.SGD([phi], lr=inner_lr)
    want = set(iters)
    out = {}
    for step in range(1, max(iters) + 1):
        opt.zero_grad()
        fitted = model(phi)                                    # (N, C)
        loss = ((fitted - target) ** 2).mean()
        loss.backward()
        opt.step()
        if step in want:
            with torch.no_grad():
                rec = model(phi)
                mse = ((rec - target) ** 2).mean().item()
                psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
                out[step] = (rec.reshape(H, W, C).clamp(0, 1).cpu(), psnr)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True,
                   help="label:checkpoint_path (repeatable).")
    p.add_argument("--sample-index", type=int, default=7)
    p.add_argument("--iters", type=str, default="3,5")
    p.add_argument("--inner-lr", type=float, default=0.01)
    p.add_argument("--data-path", type=str, default="../data")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str,
                   default="runs/diagnostics/recon_compare.png")
    args = p.parse_args()

    set_random_seeds(args.seed, args.device)
    iters = sorted({int(x) for x in args.iters.split(",") if x.strip()})
    models = [m.split(":", 1) for m in args.model]

    loader = get_cifar10_loader(args.data_path, train=False, batch_size=1,
                                num_workers=0, pin_memory=False)
    img = None
    for i, (x, _y) in enumerate(loader):
        if i == args.sample_index:
            img = x[0]                                         # (C, H, W)
            break
    if img is None:
        raise SystemExit(f"sample-index {args.sample_index} out of range")
    target_hwc = img.permute(1, 2, 0).contiguous()            # (H, W, C)

    ncols = 1 + len(iters)
    nrows = len(models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    print(f"\n=== Reconstruction PSNR (sample #{args.sample_index}) ===")
    print(f"{'model':22s} " + "  ".join(f"PSNR@{it:>3d}" for it in iters))
    for r, (label, ckpt_path) in enumerate(models):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model = build_inr(ckpt, args.device)
        snaps = fit_and_snapshot(model, target_hwc, iters, args.inner_lr, args.device)

        axes[r, 0].imshow(target_hwc.numpy())
        axes[r, 0].set_title(f"{label}\noriginal", fontsize=9)
        axes[r, 0].axis("off")
        for c, it in enumerate(iters, start=1):
            rec, psnr = snaps[it]
            axes[r, c].imshow(rec.numpy())
            axes[r, c].set_title(f"{it} iters\nPSNR {psnr:.2f} dB", fontsize=9)
            axes[r, c].axis("off")
        print(f"{label:22s} " + "  ".join(f"{snaps[it][1]:7.2f}" for it in iters))

    fig.suptitle(f"CIFAR-10 reconstruction (test sample #{args.sample_index})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(f"\nsaved figure: {args.output}")


if __name__ == "__main__":
    main()
