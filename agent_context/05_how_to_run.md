# 05 — How to run

## Environment
- Python lives in the conda env **`pss`**: `/home/omarg/miniforge3/envs/pss/bin/python`
  (this is the interpreter with torch; plain `python` may not have it).
- Git + Git-LFS binaries: `/home/omarg/miniforge3/envs/git-env/bin` (add to PATH for git ops).
- GPUs: select with `CUDA_GPU=0` / `CUDA_GPU=1` (the pipeline maps it to `CUDA_VISIBLE_DEVICES`).
- Long jobs: run inside **tmux** so they survive disconnects. Check `nvidia-smi` first.

## End-to-end pipeline (train -> makeset -> classifier -> PGD)
The main launcher is
`scripts/run_cifar10_spatial_warmstart_svd_projection_pipeline.sh`. It warm-starts
from the base e512 backbone, (optionally) trains with hard SVD projection, then
runs makeset + CNN classifier + spatial PGD.

Key env vars (defaults in parentheses):
- `EXPERIMENT` — one of: `vanilla_baseline` (matched control, no projection),
  `readout_scale`, `all_sine_readout_scale`, `all_sine_readout_modul_scale`,
  `modul_scale`, `modul_readout_scale`, and `*_L` absolute-cap variants.
- `CUDA_GPU` (0), `SVD_SCALE` (0.5, for `reference_scale` mode), `SVD_CAP` (1.0, for `_L` absolute mode).
- `NUM_EPOCHS` (5), `INNER_STEPS` (3), `MAKE_ITERS` (3).
- PGD: `PGD_STEPS` (100), `PGD_MOD_STEPS` (3), `PGD_MAX_SAMPLES` (100), `EPS_STR` ("1 2 4 6").
- Stage toggles: `RUN_TRAIN`, `RUN_MAKESET`, `RUN_CLASSIFIER`, `RUN_PGD` (all 1).
  Set `RUN_TRAIN=0` to re-run only downstream eval on an existing checkpoint.

Examples:
```bash
# Matched vanilla control on GPU 0
EXPERIMENT=vanilla_baseline CUDA_GPU=0 \
  bash scripts/run_cifar10_spatial_warmstart_svd_projection_pipeline.sh

# Readout SVD cap at 0.5x reference on GPU 1
EXPERIMENT=readout_scale CUDA_GPU=1 SVD_SCALE=0.5 \
  bash scripts/run_cifar10_spatial_warmstart_svd_projection_pipeline.sh
```
Run each variant in its own tmux window, split across the two GPUs.

## Stronger-attack runs (required before any claim — see `04`)
Scale up the same PGD entry point for serious evaluation:
- larger `PGD_MAX_SAMPLES` (>=500), more `PGD_STEPS`, more `PGD_MOD_STEPS`,
  and add restarts. Attack code: `attacks/full_pgd_cifar10_spatial.py`.
- Add AutoAttack and a transfer attack (perturb vs vanilla, apply to defended).

## Evaluation / diagnostics (session-built scripts)
Run with the `pss` python. Use `--help` for exact flags.
```bash
PY=/home/omarg/miniforge3/envs/pss/bin/python

# Reconstruction quality (PSNR) for a checkpoint
$PY scripts/reconstruct_compare.py --help

# Layer-wise adversarial amplification (||Δa_l||_2 and ratio R_l)
$PY scripts/amplification_analysis.py --help

# Verify a checkpoint enforces its SVD caps (sigma_max(W) <= cap + 1e-5)
$PY scripts/verify_svd_projection.py --help
```

## Hard SVD projection (how it's wired)
- `spectral_projection.py` builds a projection plan from `--svd-proj*` CLI flags and
  exposes `apply_projection_(plan)` which clamps singular values via exact SVD.
- `trainer.py` calls the projection **immediately after** `outer_optimizer.step()`
  (via a `post_step_fn` hook). Disabled entirely unless `--svd-proj` is set.
- SIREN detail: for sine layers the raw matrix cap is `L / omega_0`; for readout and
  modul the cap is used directly. Targets: `readout`, `pre_readout`,
  `readout_and_pre_readout`, `all_sine_readout`, `all_sine_readout_modul`, `modul`,
  `modul_readout`.

## Git / pushing artifacts
- Code pushes work normally. **Large LFS artifacts currently do NOT push** (GitHub
  LFS budget exceeded). New `.pth` are untracked-on-disk; functasets are git-ignored.
- To push artifacts later: increase the LFS data pack, then:
  ```bash
  export PATH="/home/omarg/miniforge3/envs/git-env/bin:$PATH"
  git add -A && git commit -m "add checkpoints/functasets (LFS)" && git push origin main
  ```
- Never commit secrets/tokens. Never edit a running pipeline `.sh`. Keep `.sh` as LF.
