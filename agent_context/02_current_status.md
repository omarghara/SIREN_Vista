# 02 — Current status (as of 2026-06-25)

## Where we are
CIFAR-10 reconstruction is good; clean classifier top-1 is ~76% (inner-5 matched).
We have tried several weight-space regularizers to improve PGD robustness:
soft-Lipschitz caps, warm-start cap sweeps, counter-amplification, orthogonality,
and most recently **hard SVD spectral projection**. **None has produced a clean,
convincing, attack-validated robustness win yet.**

## Latest experiment family: hard SVD projection (inner-3, warm-started from vanilla)
Protocol for this family: 5 meta-epochs, warm-started from the vanilla e512
backbone, then **inner-3** makeset + CNN classifier + PGD (100 steps, n=100,
eps {1,2,4,6}/255, mod-steps 3). The **matched control** is
`warmvanilla_baseline_e5` (same protocol, no projection).

PGD-100, n=100 (robust acc):

| model | clean | eps1 | eps2 | eps4 | eps6 |
|---|---:|---:|---:|---:|---:|
| warmvanilla_baseline_e5 (CONTROL) | 0.830 | 0.580 | 0.330 | 0.040 | 0.010 |
| svdproj_readout_scale0.5_e5 | 0.820 | 0.500 | 0.310 | 0.070 | 0.010 |
| svdproj_modul_scale0.5_e5 | 0.830 | 0.520 | 0.300 | 0.070 | 0.010 |
| svdproj_modul_readout_scale0.7_e5 | 0.780 | 0.530 | 0.320 | 0.050 | 0.010 |
| svdproj_all_sine_readout_scale0.5_e5 | 0.770 | 0.570 | 0.330 | 0.080 | 0.020 |

Read this honestly: differences are within noise at n=100. The only mild,
consistent signal is a tiny bump at **eps4** for the SVD variants (0.07–0.08 vs
0.04 control), but clean accuracy also dropped for the stronger caps, and
everything collapses by eps6. **Not a result.** n=100 is too small to conclude;
binomial SE at 0.05 is ~2.2pp.

`all_sine_readout` (the "whole SIREN" cap) also reconstructs noticeably worse —
expected from the math (see `03`), capping every sine layer kills expressivity.

## Older context (inner-5 family, larger n)
`context/attack_currenct_results.md` has the inner-5 PGD-200 sweeps (n up to 1000)
for vanilla e512 vs softlip-tiered and the warm-start regularizer sweeps. Summary:
softlip had at most a small edge at eps {1,2,4}/255 and both collapse by eps {6,8}.
Do **not** mix inner-3 and inner-5 numbers — different fitting budgets.

## Current checkpoints (model_cifar10/)
- `functa_like_cifar10_..._e512_inner3_...` — **base backbone**; everything warm-starts from it. Do not delete.
- `cifar10_spatial_warmvanilla_warmvanilla_baseline_e5` — matched inner-3 vanilla control.
- `cifar10_spatial_warmvanilla_svdproj_{readout_scale0.5, modul_scale0.5, modul_readout_scale0.7, all_sine_readout_scale0.5}_e5` — SVD variants.
- `cifar10_spatial_warmsoftlip_*` and other `warmvanilla_*` — earlier regularizer runs.
Run outputs live under `runs/cifar10_spatial_svd_projection/` and `runs/cifar10_spatial_inner5_*`.

## IMPORTANT: repo / artifact state
- The repo is Git + Git-LFS (`.pth`, `runs/**/*.pkl` tracked via LFS). Remote:
  `github.com/omarghara/SIREN_Vista` (origin is SSH).
- **GitHub LFS budget is exceeded** — the remote currently rejects ANY new LFS
  upload. Because of this, the last commit pushed **code only**. These are kept
  **local-only, not on the remote**:
  - new model/run `.pth` (~99 MB) — left untracked on disk, ready to commit once budget is restored.
  - functaset `.pkl` caches (~2.3 GB) — git-ignored via `runs/**/functaset/*.pkl`.
- So if you clone fresh from GitHub you will NOT get the new checkpoints/functasets.
  To push them later: increase the LFS data pack, then `git add -A && commit && push`.
- A working `git` (with LFS) is at `/home/omarg/miniforge3/envs/git-env/bin`.

## Known gotchas
- Shell scripts must be **LF** (CRLF breaks the shebang). `.gitattributes` enforces `*.sh eol=lf`.
- Don't edit a pipeline `.sh` while it's running (bash reads by byte offset and corrupts mid-run).
- SVD cap verification can show ~1e-5 "FAIL" deltas — that's float32 round-off, not a bug.
- There were `*.cpython-310.py_failed` files in the root from a past recovery; the live source is the plain `.py`.
