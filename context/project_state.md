# Project state — context for future chats

A single-page orientation for any new Cursor chat that picks up this thesis work. Read `cursor_context.md` first for the scientific framing and `chatgpt_deep_search.md` for the deep literature review. This file is the practical status board.

---

## 1. What this project is

Master's thesis on **adversarial robustness of parameter-space classifiers built on SIREN INRs**. The pipeline:

```
signal x  →  inner INR fit  →  modulation vector z*(x)  →  classifier g(z*(x))  →  prediction
```

Upstream paper the thesis builds on: *Adversarial Attacks in Weight-Space Classifiers* (Shor et al., arXiv:2502.20314). Its central claim is that observed robustness of parameter-space classifiers is largely **gradient obfuscation from the inner fitting loop**, not structural robustness. The thesis goal is to replace that accidental robustness with a principled one. See the ideas catalog for directions.

## 2. Filesystem layout on this machine

All paths are absolute.

```
/home/omarg/
├── context/                          # scientific + handoff context
│   ├── cursor_context.md             # high-level project brief (read first)
│   ├── chatgpt_deep_search.md        # long-form literature + math reference
│   ├── CHANGES.md                    # changelog of edits made during chats
│   └── project_state.md              # THIS FILE
├── ideas/
│   └── robustness_ideas.md           # 24 ranked research ideas (categories A–J)
├── SIREN_Vista/                      # local working copy — edit here
│   ├── SIREN.py                      # ModulatedSIREN, ModulatedSIREN3D, SineAffine
│   ├── trainer.py                    # meta-training (+ resume + variants + sigma log)
│   ├── makeset.py                    # functaset creation (+ variants)
│   ├── train_classifier.py           # downstream classifier (variant-agnostic)
│   ├── evaluate_reconstruction.py    # MSE/PSNR/SSIM vs inner-loop iters
│   ├── diagnostics.py                # per-layer spectral-norm helpers
│   ├── dataloader.py                 # MNIST / Fashion-MNIST loaders + functaset
│   ├── dataloader_modelnet.py        # ModelNet10
│   ├── utils.py                      # seeding, LR schedule, metrics
│   ├── attacks/                      # attack suite (not yet modified)
│   ├── scripts/                      # end-to-end pipeline scripts
│   │   └── run_soft_lipschitz_mnist.sh
│   └── variants/                     # ***variant plug-in registry***
│       ├── __init__.py
│       ├── vanilla.py                # no-op default
│       └── soft_lipschitz.py         # first real variant
├── Parameter-Space-Attack-Suite/     # pristine upstream clone, do not edit
└── data/                             # datasets
```

The `SIREN_Vista/` directory is the working fork; `Parameter-Space-Attack-Suite/` is the upstream repo and should stay clean for diffing.

## 3. Python environment

```
~/miniforge3/envs/pss/bin/python
```

- Python 3.10, torch 2.0.1 + CUDA 11.7, CUDA available.
- `~/miniforge3/envs/pss` is the conda env named `pss`. It has torch and the auto-attack fork pinned by the upstream `environment.yml`.
- Never use `/usr/bin/python3` — no torch there.

## 4. What has already been built

### Resume-from-checkpoint (`trainer.py`)

- `--resume PATH` restores model, optimizer, epoch counter, and `best_loss`. Old checkpoints without `optimizer_state_dict` fall back to a fresh optimizer with a warning.
- Epoch loop semantics: `--num-epochs N` means "N more epochs" when resuming.
- Saved payload always includes `optimizer_state_dict`.

### Variants plug-in system (`variants/` package)

A small registry-based plug-in system for research variants. Each variant is one Python file with a class decorated `@register("name")` exposing four static methods:

| method | purpose |
|---|---|
| `add_args(parser)` | register CLI flags, prefixed with the variant name |
| `build(base_model, args)` | optionally wrap / modify the base `ModulatedSIREN` |
| `penalty(model, args)` | auxiliary training loss (0 for a pure-architecture variant) |
| `slug(args)` | short identifier used for the save subdir |

The trainer passes the penalty to `fit()` via a callback and saves to `model_{dataset}/{slug}/modSiren.pth`. Checkpoints also record `variant`, `variant_args`, `model_args` for provenance.

**Variants currently shipped:**

- **`vanilla`** — default, reproduces original training. Slug `"vanilla"`.
- **`soft_lipschitz`** — adds `λ · Σ_l max(0, σ(W_l) − c_l)^2` over selected linear layers, with **per-layer cap `c_l` derived from a single per-layer Lipschitz budget `L = --soft-lip-cap`**:
    - sine hidden layers → `c_l = L / freq` (so `freq · σ(W) ≤ L`),
    - readout `hidden2rgb` → `c_l = L` (pure linear, no sine),
    - `modul` (mode `all`) → `c_l = L / freq` (feeds into a sine).

    Spectral norm is estimated by **differentiable power iteration with persistent u/v buffers** (mirrors `torch.nn.utils.spectral_norm`'s pattern but does not rescale W). CLI knobs:
    - `--soft-lip-cap L` (default 1.0 — per-layer Lipschitz budget)
    - `--soft-lip-lambda λ` (default 1e-2)
    - `--soft-lip-apply-to {sine_only, sine_and_readout, all}` (default `sine_only`)
    - `--soft-lip-power-iters K` (default 1)

    Slug: `softlip_L{L:g}_lam{λ:.0e}_{apply_to}`, e.g. `softlip_L1_lam1e-02_sine_only`. See `CHANGES.md` §7 for the full motivation and a calibration note (`L=1` is tight; for pre-change behaviour use `L=30`).

### `makeset.py` accepts `--variant`

Needed only because future variants may change the state-dict shape (e.g. add `weight_orig` / `weight_u` / `weight_v` buffers from `torch.nn.utils.spectral_norm`). For `vanilla` and `soft_lipschitz` this is a no-op.

### Reconstruction evaluation (`evaluate_reconstruction.py`)

Standalone script that **re-fits** modulations on train/test and reports MSE, PSNR, SSIM as a function of inner-loop iter count (default `--iter-checkpoints "5,20,50,100,200"`), in one pass through the data. Hand-rolled SSIM (no `scikit-image`/`torchmetrics` dep). Output: JSON next to the checkpoint + stdout table.

- Re-fit rather than reuse stored functaset because `makeset.split()` shuffles entries, so image-to-modulation correspondence is lost in the pkl.
- The `iters=5` row is apples-to-apples with what the downstream classifier sees when `makeset.py --iters 5` was used; higher rows approximate backbone capacity, matching Functa-paper fitting curves.
- Wired as Step 4/4 of [SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh](SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh); bash knobs: `EVAL_ITERS`, `EVAL_MAX_SAMPLES` (default 2000 per split — set blank for full eval).
- See `CHANGES.md` §8 for the full design rationale, citing [SIREN paper](https://arxiv.org/abs/2006.09661) and [Functa paper](https://arxiv.org/abs/2201.12204).

### Verification

Smoke-tested on CPU with a small `ModulatedSIREN(depth=4)`: vanilla penalty = 0, soft-Lipschitz penalty flows gradients to every `SineAffine.affine.weight`, layer selection hits 4 / 5 / 6 for the three modes, slugs format correctly. `--help` on both trainer and makeset shows the variant arg group cleanly. `evaluate_reconstruction.py` smoke-tested on 8 MNIST samples against a synthetic tiny checkpoint: produces PSNR ~10 dB (random-output baseline), monotonically decreasing MSE across iter snapshots, and a well-formed JSON summary.

## 5. What has NOT been built yet

### Robustness directions (from `/home/omarg/ideas/robustness_ideas.md`)

Top near-term candidates (user's first wave):

1. **Spectral-normalized SIREN** — *hard* Lipschitz constraint via `torch.nn.utils.spectral_norm`. This is the natural next variant (`variants/hard_lipschitz.py`). Unlike soft-Lipschitz, it modifies the state-dict shape, so `variants.build(...)` will no longer be a no-op and `makeset.py`'s variant-aware loading becomes load-bearing.
2. **Modulation-stability regularizer** — penalizes `‖z*(x) − z*(x+δ)‖` on the fitted modulation. Attacks the *actual* target of robustness, not just the backbone. Requires either a second inner-loop rollout per batch or implicit differentiation.
3. **Amortized / hypernetwork INR fitting** — *diagnostic*, not a defense. If apparent robustness vanishes once the iterative inner loop is replaced by a single forward pass, that alone confirms the obfuscation story.
4. **Modulation-space TRADES** — cheap adversarial training in z-space, stackable on top of any SIREN-side defense.

See `ideas/robustness_ideas.md` for the full 24-idea catalog organized into 10 categories.

### Evaluation infrastructure

No adaptive-attack evaluation has been plumbed yet. The `attacks/` directory is still stock upstream. Robustness claims must not be made before:

- naive PGD baseline,
- unrolled-through-fit PGD,
- implicit-differentiation attack,
- BPDA-style approximation,
- at least one black-box (Square or NES) sanity check,
- the attack suite from Shor et al.

### Ablations / diagnostics missing

- ~~No logging of per-layer `σ_max(W_l)` during training.~~ *Addressed by [SIREN_Vista/diagnostics.py](SIREN_Vista/diagnostics.py) + `--log-sigmas-every N` on `trainer.py`; see `CHANGES.md` §10.*
- ~~No reconstruction-fidelity comparison (vanilla vs. soft-Lipschitz) on held-out signals.~~ *Addressed by `evaluate_reconstruction.py` (§4). Still need a side-by-side comparison script / notebook that overlays two `reconstruction_eval.json` files.*
- No representation-shift curves `‖z*(x) − z*(x+δ)‖` vs. `‖δ‖`.

## 6. Conventions (apply when editing)

- **Hard rule from `cursor_context.md`**: never claim robustness from naive PGD alone. Every defense must be stress-tested adaptively. If you change the SIREN, you must also think about whether your change makes gradients harder to compute — that would be obfuscation, not robustness.
- **Condition number ≠ Lipschitz**. Controlling `κ(W)` does not bound amplification. Always control `σ_max` explicitly if robustness is the goal.
- **Saved checkpoint is the source of truth for provenance.** When adding a variant, always record its hyperparameters in `variant_args` so experiments are reproducible from the `.pth` alone.
- **Adding a variant is a 4-step recipe** (see §12 of `CHANGES.md`).
- **Do not edit `Parameter-Space-Attack-Suite/`**. Use `SIREN_Vista/` as the editable tree.

## 7. Fast commands reference

```bash
PY=~/miniforge3/envs/pss/bin/python

# Train vanilla for 6 epochs on MNIST
$PY trainer.py --dataset mnist --num-epochs 6

# Train soft-Lipschitz (λ=1e-2, c=1.0, sine-only)
$PY trainer.py --dataset mnist --num-epochs 6 --variant soft_lipschitz \
    --soft-lip-cap 1.0 --soft-lip-lambda 1e-2

# Resume the soft-Lipschitz run for 4 more epochs
$PY trainer.py --dataset mnist --num-epochs 4 --variant soft_lipschitz \
    --soft-lip-cap 1.0 --soft-lip-lambda 1e-2 \
    --resume model_mnist/softlip_L1_lam1e-02_sine_only/modSiren.pth

# Build the functaset (flags must match the trained variant)
$PY makeset.py --dataset mnist --variant soft_lipschitz \
    --soft-lip-cap 1.0 --soft-lip-lambda 1e-2 \
    --checkpoint model_mnist/softlip_L1_lam1e-02_sine_only/modSiren.pth

# Train the classifier (variant-agnostic, works on any functaset .pkl)
$PY train_classifier.py --dataset mnist \
    --functaset-path-train functaset/mnist_train.pkl \
    --functaset-path-test functaset/mnist_test.pkl

# Reconstruction eval (MSE/PSNR/SSIM vs inner-loop iters; variant flags must match)
$PY evaluate_reconstruction.py --dataset mnist \
    --checkpoint model_mnist/softlip_L1_lam1e-02_sine_only/modSiren.pth \
    --variant soft_lipschitz --soft-lip-cap 1.0 --soft-lip-lambda 1e-2 \
    --iter-checkpoints "5,20,50,100,200" \
    --split both --max-samples 2000

# Full end-to-end soft-Lipschitz pipeline (trainer + makeset + classifier + eval)
bash ~/SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh
```

## 8. Useful references inside this repo

- The plan file for the resume feature: `.cursor/plans/resume_training_in_siren_trainer_*.plan.md`.
- The plan file for the variants infrastructure: `.cursor/plans/siren_variants_infrastructure_plus_soft_lipschitz_*.plan.md`.
- Full change log from this session: `context/CHANGES.md`.
- Robustness idea catalog: `ideas/robustness_ideas.md`.
- Thesis-level scientific brief: `context/cursor_context.md`.
- Long-form literature + math background: `context/chatgpt_deep_search.md`.

Future chats should read this file plus `cursor_context.md` and the most recent entries of `CHANGES.md` before proposing edits.
