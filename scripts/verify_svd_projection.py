#!/usr/bin/env python
"""Verify that a checkpoint's projected layers satisfy sigma_max(W) <= cap + tol.

Usage
-----
# Read caps from the checkpoint's saved projection_args:
python scripts/verify_svd_projection.py \
    --checkpoint model_cifar10/<run>/modSiren.pth

# Or check a single explicit cap against one weight key:
python scripts/verify_svd_projection.py \
    --checkpoint model_cifar10/<run>/modSiren.pth \
    --state-key siren.hidden2rgb.weight --cap 1.0

Exit code is 0 when every checked layer is within cap + tol, else 1.
"""

import argparse
import os
import sys

# Allow running from anywhere: add repo root (parent of this scripts/ dir).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import spectral_projection  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to modSiren.pth.")
    p.add_argument("--state-key", default=None,
                   help="Optional single weight key to check (with --cap).")
    p.add_argument("--cap", type=float, default=None,
                   help="Cap to use with --state-key (overrides saved caps).")
    p.add_argument("--tol", type=float, default=1e-5, help="Tolerance (default 1e-5).")
    args = p.parse_args()

    caps = None
    if args.state_key is not None or args.cap is not None:
        if args.state_key is None or args.cap is None:
            p.error("--state-key and --cap must be given together.")
        caps = {args.state_key: args.cap}

    ok, rows = spectral_projection.verify_checkpoint(
        args.checkpoint, caps=caps, tol=args.tol,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"{'layer':40s} {'cap':>12s} {'sigma_max':>12s}  status")
    print("-" * 80)
    for r in rows:
        sigma_str = "n/a" if r["sigma"] is None else f"{r['sigma']:.6g}"
        status = "OK" if r["ok"] else "FAIL"
        note = f"  ({r['note']})" if r["note"] else ""
        print(f"{r['layer']:40s} {r['cap']:>12.6g} {sigma_str:>12s}  {status}{note}")
    print("-" * 80)
    print("ALL WITHIN CAP" if ok else "CAP VIOLATION DETECTED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
