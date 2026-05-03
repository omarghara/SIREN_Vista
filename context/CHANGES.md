# Recent changes to `SIREN_Vista`

Chronological log of edits made to the local `SIREN_Vista` copy of the Parameter-Space-Attack-Suite during this working session. All paths are relative to `/home/omarg/`.

---

## 1. `trainer.py` — resume-from-checkpoint

**File.** `SIREN_Vista/trainer.py`.

Added a `--resume PATH` CLI flag and a checkpoint-loading block that restores, from a saved `.pth`:

- `state_dict` — model weights.
- `optimizer_state_dict` — Adam optimizer state (only when present; old files without it just start with a fresh optimizer and print a warning).
- `epoch` — continues the counter. New loop: `range(start_epoch, start_epoch + args.num_epochs)`. `--num-epochs` now means *this many additional epochs* when resuming.
- `loss` — kept as `best_loss`, so the "only save when better" heuristic keeps working across resumes.

The `torch.save` payload now also writes `optimizer_state_dict`. Old checkpoints still load fine (graceful fallback).

## 2. `variants/` package — research-variant infrastructure

**New files.** `SIREN_Vista/variants/{__init__.py, vanilla.py, soft_lipschitz.py}`.

Small plugin-style registry so every research idea (Lipschitz variants, WIRE, FINER, modulation-stability, etc.) can be added as one new file without touching the trainer.

### Public API (`variants/__init__.py`)

```
variants.REGISTRY               # dict[name -> class]
variants.register(name)         # class decorator
variants.available()            # sorted list of names
variants.get(name)              # class
variants.add_all_variant_args(parser)   # register every variant's CLI flags
variants.build(name, base_model, args)  # optional architecture modification
variants.penalty(name, model, args)     # training-time auxiliary loss
variants.slug(name, args)               # run identifier for savedir
variants._extract_variant_args(args, name)   # dict of ns -> compact metadata
```

Each variant is a class decorated with `@register("name")` providing four static methods: `add_args`, `build`, `penalty`, `slug`.

### Variants shipped now

- **`vanilla`** (default). `add_args` no-op, `build` identity, `penalty` returns a zero tensor, slug = `"vanilla"`.
- **`soft_lipschitz`**. Adds a training-time penalty
    `λ · Σ_l max(0, σ(W_l) − c_l)^2`
  over selected linear layers, with σ = ‖W_l‖_2 estimated via **persistent-buffer power iteration** (`_sl_u`, `_sl_v` lazily attached to each targeted `nn.Linear`; same pattern as `torch.nn.utils.spectral_norm` but **without** rescaling the weights). `σ = uᵀ W v` is differentiable w.r.t. W because u, v are updated under `torch.no_grad()`. CLI flags:

    | flag | default | meaning |
    |---|---|---|
    | `--soft-lip-cap` | `1.0` | per-layer Lipschitz budget **L** (see §7 for per-layer cap derivation) |
    | `--soft-lip-lambda` | `1e-2` | penalty weight λ |
    | `--soft-lip-apply-to` | `sine_only` | `sine_only` / `sine_and_readout` / `all` |
    | `--soft-lip-power-iters` | `1` | power-iteration steps per batch |

  Layer-selection sizes on a `ModulatedSIREN(depth=4)`: 4 / 5 / 6 (SineAffine.affine → + `hidden2rgb` → + `modul`).

  Slug: `softlip_L{cap:g}_lam{lambda:.0e}_{apply_to}`, e.g. `softlip_L1_lam1e-02_sine_only`. *(Updated from `c` prefix in §7.)*

## 3. `trainer.py` — variant wiring

Further updates on top of §1:

- New CLI: `--variant {vanilla,soft_lipschitz}` (default `vanilla`) and `--run-name NAME`. Every variant's own flags are appended via `variants.add_all_variant_args(parser)`, so `--help` shows all knobs.
- After base model construction: `modSiren = variants.build(args.variant, modSiren, args)`. No-op today for both shipped variants; reserved for future variants that change architecture (e.g. `spectral_norm`-wrapped hard-Lipschitz).
- `fit()` gained a `penalty_fn=None` keyword arg. When set, `penalty_fn(model)` is added to `outer_loss` **before** `.backward()`, so the variant penalty participates in gradient updates. Trainer passes `penalty_fn = lambda m: variants.penalty(args.variant, m, args)`.
- Save path is now `model_{dataset}/{run_slug}/modSiren.pth`, where `run_slug = args.run_name or variants.slug(args.variant, args)`. With default flags: `model_mnist/vanilla/modSiren.pth`, `model_mnist/softlip_L1_lam1e-02_sine_only/modSiren.pth`, etc.
- Checkpoint payload adds:
  - `variant` — the variant name used during training.
  - `variant_args` — dict of only the CLI flags belonging to that variant.
  - `model_args` — `{dataset, hidden_dim, mod_dim, depth}`.
- Resume block now warns (does not abort) if the resumed `ckpt['variant']` differs from the current `--variant`.

## 4. `makeset.py` — variant-aware

**File.** `SIREN_Vista/makeset.py`.

- New CLI: `--variant` + every variant's flags (via `variants.add_all_variant_args(parser)`).
- Calls `modSiren = variants.build(args.variant, modSiren, args)` **before** `load_state_dict(...)`. Identity for `vanilla` and `soft_lipschitz`; reserved so future architecture-modifying variants can load cleanly.

Downstream classifier training (`train_classifier.py`) is untouched — it operates on modulation vectors, which are variant-agnostic.

## 5. Verification performed in this session

- `trainer.py --help` / `makeset.py --help` render cleanly with the `soft_lipschitz:` argument group.
- A CPU smoke test constructs a small `ModulatedSIREN(depth=4)`, exercises every registered variant, confirms:
    - vanilla penalty is exactly 0 and detached,
    - soft-Lipschitz `all` with tight cap produces a strictly positive penalty,
    - `.backward()` produces a non-zero gradient on every `SineAffine.affine.weight`,
    - layer-selection sizes (4/5/6) match expectations,
    - slug formatting produces paths like `softlip_c1_lam1e-02_sine_only`.

No automated test suite; the smoke test was run ad-hoc with `~/miniforge3/envs/pss/bin/python`.

## 6. Example commands

```bash
# Vanilla MNIST training (saves to model_mnist/vanilla/modSiren.pth):
~/miniforge3/envs/pss/bin/python trainer.py --dataset mnist --num-epochs 6

# Soft-Lipschitz training:
~/miniforge3/envs/pss/bin/python trainer.py --dataset mnist --num-epochs 6 \
    --variant soft_lipschitz --soft-lip-cap 1.0 --soft-lip-lambda 1e-2

# Resume the soft-Lipschitz run for 4 more epochs:
~/miniforge3/envs/pss/bin/python trainer.py --dataset mnist --num-epochs 4 \
    --variant soft_lipschitz --soft-lip-cap 1.0 --soft-lip-lambda 1e-2 \
    --resume model_mnist/softlip_L1_lam1e-02_sine_only/modSiren.pth

# Build the functaset (must pass the same --variant and flags):
~/miniforge3/envs/pss/bin/python makeset.py --dataset mnist \
    --variant soft_lipschitz --soft-lip-cap 1.0 --soft-lip-lambda 1e-2 \
    --checkpoint model_mnist/softlip_L1_lam1e-02_sine_only/modSiren.pth
```

## 7. `soft_lipschitz` — cap re-interpreted as per-layer Lipschitz budget L

**File.** `SIREN_Vista/variants/soft_lipschitz.py` (later edit in the same session).

Same penalty *shape* (`λ · Σ max(0, σ(W_l) − c_l)²`) but the per-layer cap `c_l` is now **derived** from a single user-provided per-layer Lipschitz budget `L = args.soft_lip_cap`, instead of all layers sharing one raw σ-cap.

| layer | layer map | Lipschitz | σ-cap enforced |
|---|---|---|---|
| `SineAffine.affine` (hidden sine) | `sin(freq·(Wx + b + s))` | `freq · σ(W)` | **`L / freq`** |
| `SIREN.hidden2rgb` (readout) | `Wx + b` | `σ(W)` | **`L`** |
| `ModulatedSIREN(.3D).modul` (mode `all`) | shift → fed into sine | `freq · σ(W)` | **`L / freq`** |

**Why.** Previously every layer shared the same σ-cap `c`, which implicitly gave sine layers a `freq`× larger *Lipschitz* budget than the readout. Now all penalized layers have the same **target Lipschitz `L`** regardless of whether they sit before a sine activation. `freq` is read from each `SineAffine` directly, so non-default `freq` still works; fallback `30.0` is used for `modul` if no `SineAffine` is present.

### Code changes

- `_collect_layers(model, mode)` → `_collect_layers(model, mode, L)` and returns a list of `(nn.Linear, sigma_cap)` pairs instead of plain layers.
- `SoftLipschitz.penalty` iterates over `(lin, cap)` pairs applying the **per-layer** cap inside `max(0, σ − cap)²`.
- `SoftLipschitz.slug` prefix changed from `c{…}` to `L{…}` (e.g. `softlip_L1_lam1e-02_sine_only`) so old and new runs are visually distinguishable on disk.
- `--soft-lip-cap` help text updated. Module docstring rewritten to document the per-layer derivation.
- No changes elsewhere (`trainer.py`, `makeset.py`, `variants/__init__.py`, other variants) — the function signature of `SoftLipschitz.penalty` is unchanged and `_extract_variant_args` still picks up `soft_lip_*` by prefix.

### Behavioural differences (calibration heads-up)

- Default `--soft-lip-cap 1.0` now means **L = 1 per-layer Lipschitz**, i.e. sine layers are capped at σ ≈ 0.033 (very tight) at `freq = 30`. Expect the penalty to be far more aggressive than before with the same numeric value. For behaviour closest to the pre-change defaults (σ-cap ≈ 1 on sine layers), pass `--soft-lip-cap 30.0`.
- Save directories gain an `L` prefix; old `softlip_c*` directories remain untouched.
- Old `.pth` files store `variant_args.soft_lip_cap` under the old meaning (raw σ-cap). The code does not reinterpret it, but be careful when comparing numerically across eras.

### Verification (CPU smoke test, `ModulatedSIREN(hidden=16, depth=3)`, `L=2`)

- sine caps = 2/30 ≈ 0.0667 (three layers).
- `hidden2rgb` cap = 2.0.
- `modul` cap (mode `all`) = 2/30 ≈ 0.0667.
- Penalty nonzero and `requires_grad=True`; after `.backward()`, `grad_norm > 0` on every `SineAffine.affine.weight` and on `modul`. `hidden2rgb` is below its (loose) cap at init so its grad is 0, as expected.
- Slug: `softlip_L2_lam1e-02_all`.

## 8. `evaluate_reconstruction.py` — reconstruction-quality eval

**New file.** `SIREN_Vista/evaluate_reconstruction.py`.

Measures reconstruction fidelity of the trained SIREN backbone on train and/or test splits, reporting **MSE / PSNR / SSIM as a function of inner-loop iteration count**, in one pass through the data. Mirrors the fitting-curve style of:

- SIREN (Sitzmann et al. 2020, [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)) — PSNR vs step count.
- Functa (Dupont et al. 2022, [arXiv:2201.12204](https://arxiv.org/abs/2201.12204)) — per-signal reconstruction PSNR after inner-loop fit.

### Why re-fit instead of loading stored modulations

The functaset produced by `makeset.py` stores only `{'modul': <np.ndarray>, 'label': int}` per entry, and `split()` shuffles entries via `torch.randperm` before pickling (see [SIREN_Vista/makeset.py:91-107](SIREN_Vista/makeset.py)). Image-to-modulation correspondence is therefore unrecoverable from the pkl alone. The eval script re-fits modulations from zero-init with the same optimizer and lr as makeset (`optim.SGD(lr=0.01)` for MNIST, `optim.Adam` for voxels, or `optim.LBFGS` if `--lbfgs`), so:

- The **leftmost** entry of `--iter-checkpoints` should match makeset's `--iters` so its row is apples-to-apples with what the downstream classifier sees.
- The **rightmost** entry (default 200) probes the backbone's expressive ceiling (saturated reconstruction).

### Inner-loop snapshotting

Implemented in [SIREN_Vista/evaluate_reconstruction.py](SIREN_Vista/evaluate_reconstruction.py) as:

```python
for step in range(1, max_step + 1):
    optimizer.zero_grad()
    loss = MSE(model(modulator), image)
    loss.backward()
    optimizer.step()
    if step in step_set:
        with torch.no_grad():
            snapshots[step] = (mse, psnr, ssim)
```

This visits each image exactly once and records metrics at every requested step count inside the same fit trajectory — far cheaper than restarting the inner loop per checkpoint. Backbone parameters are `requires_grad_(False)` so no unnecessary autograd overhead on the model side.

For `--lbfgs` the script falls back to independent fits per checkpoint (LBFGS history cannot be shared across `optimizer.step(closure)` calls without losing the quasi-Hessian buffer), and documents this in its `--help`.

### SSIM

Hand-rolled in the same file as `ssim_2d(fitted, target)`: 11-tap Gaussian window via `torch.nn.functional.conv2d`, constants `C1 = 0.01^2`, `C2 = 0.03^2`, data_range fixed to 1.0 (MNIST is `T.ToTensor()` which yields [0, 1]). Zero external deps — chose this over `scikit-image`/`torchmetrics` because the `pss` conda env currently lacks both and we want the script self-contained. For voxel datasets SSIM is skipped and the JSON simply omits the `ssim` block.

### CLI

```
--checkpoint PATH         (required)
--dataset {mnist,fmnist,modelnet}  (required)
--variant NAME + --soft-lip-* ...  (must match the checkpoint; passed through variants.add_all_variant_args)
--split {train,test,both}          (default both)
--iter-checkpoints "5,20,50,100,200"
--inner-lr 0.01      --lbfgs (flag)
--max-samples N      (default: full split; set for quick runs)
--output PATH        (default: <checkpoint_dir>/reconstruction_eval.json)
--hidden-dim/--mod-dim/--depth (auto-override from ckpt['model_args'] when present)
```

### Output

Per split, stdout ASCII table:

```
=== Reconstruction eval: test (n=10000) ===
 iters         MSE   PSNR mean/med   SSIM mean
     5   2.41e-02   16.18 / 16.05      0.6200
    20   6.50e-03   21.87 / 22.01      0.8300
    ...
```

JSON saved to `--output` with full provenance:

```json
{
  "checkpoint": "...",
  "variant": "soft_lipschitz",
  "variant_args": {...},
  "model_args": {...},
  "iter_checkpoints": [5, 20, 50, 100, 200],
  "max_samples": 2000,
  "results": {
    "train": {"n_samples": N, "at_iters": {"5": {"mse": {"mean": .., "median": .., "std": .., "min": .., "max": ..}, "psnr": {...}, "ssim": {...}}, "20": {...}}},
    "test":  {...}
  }
}
```

### Smoke test

Synthesized a tiny checkpoint (hidden=32, depth=3, mod=64) with the expected `model_args`/`variant_args` schema and ran the script on CPU over 8 MNIST samples per split with `--iter-checkpoints "5,20,50"`. Confirmed:

- CLI parses, variant group registers its flags, checkpoint loads, `model_args` override prints.
- Per-image inner loop runs, snapshots populated at every requested step, SSIM returns finite in [0, 1].
- Untrained SIREN gives PSNR ~ 9-11 dB and MSE ~ 0.1 (as expected; random output on [0, 1] MNIST).
- MSE decreases across snapshot steps (0.119 → 0.115) confirming the SGD loop advances.
- JSON schema populated exactly as specified.

### Wired into the bash pipeline

[SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh](SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh) now has a **Step 4/4** that calls the eval script with the same variant flags as Steps 1-2. Two new top-of-script knobs: `EVAL_ITERS` (default `"5,20,50,100,200"`) and `EVAL_MAX_SAMPLES` (default `2000` per split). Output lands at `${RUN_ROOT}/reconstruction_eval.json`. Leaving `EVAL_MAX_SAMPLES` blank runs the full split (minutes on GPU).

## 9. `trainer.py` — MSE vs penalty split in tqdm + epoch log

**File.** [SIREN_Vista/trainer.py](SIREN_Vista/trainer.py) `fit()` inner loop.

Previously the tqdm description printed one opaque number: `outer_loss = MSE + λ·Σ max(0, σ − c_l)²`. When tuning soft-Lipschitz hyperparameters this hides whether the loss floor is coming from reconstruction or from the regularizer. Now the outer-step block:

- Caches `mse_component = outer_loss.detach().item()` **before** adding the penalty.
- Caches `pen_component = pen.detach().item()` when `penalty_fn` is set (0.0 otherwise, so vanilla is unaffected).
- Both are appended to per-epoch running lists `mse_losses`, `pen_losses` (alongside the existing `losses`).
- tqdm description is now `"Epoch N | total 0.33 | MSE 0.30 | pen 0.03"`.
- End-of-epoch line now reads `"epoch: N, total: 0.32, MSE: 0.29, pen: 0.03"`.

`fit()` still returns the average *total* loss so the `best_loss`-guarded save path in `__main__` is unchanged. Vanilla runs just see `pen=0.0000` everywhere — no behavioural change, only more readable logs.

Motivation: during the first end-to-end soft-Lipschitz run at `L=1, apply_to=all`, total loss stuck at ~0.33. Without the split it was ambiguous whether that was MSE ~0.30 + penalty ~0.03 (reconstruction choked by over-tight caps on `modul` and first sine) or the reverse. The split immediately reveals which knob to move.

## 10. `diagnostics.py` — per-layer spectral-norm monitoring

**New file.** [SIREN_Vista/diagnostics.py](SIREN_Vista/diagnostics.py).

Standalone helper (deliberately not part of the `variants/` package) exposing:

- `layer_sigmas(model, n_iter=30)` → `OrderedDict[str, float]` with entries `sine.0`, `sine.1`, ..., `readout`, `modul`. Uses fresh-buffer random-init power iteration (independent of the soft-Lipschitz variant's persistent buffers), so it works for any model and any variant.
- `format_sigmas_one_liner(sigmas)` → compact one-line string `"sine[10]: [... ...] min=0.093 max=4.982 | readout: 0.067 | modul: 1.861"`.

Threaded into training via:

- New flag `--log-sigmas-every N` on [SIREN_Vista/trainer.py](SIREN_Vista/trainer.py) (default 0 = off).
- When `> 0`, inside `fit()` after each `outer_optimizer.step()` where `batch_idx % N == 0`, a line is emitted via `tqdm.write` (preserves the progress bar).
- New top-of-script knob `LOG_SIGMAS_EVERY=50` in [SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh](SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh), passed through as `--log-sigmas-every`.

### Calibration data from the upstream vanilla checkpoint

Running `layer_sigmas(...)` on `/home/omarg/Parameter-Space-Attack-Suite/model_mnist/modSiren.pth` produced:

| layer | σ (trained vanilla) |
|---|---|
| sine.0 (first, in=2) | **4.982** — essentially unchanged from init (4.98), effectively frozen |
| sine.1 – sine.5 | 0.093 – 0.096 — barely moved from init (~0.09) |
| sine.6 – sine.9 | 0.104 – 0.124 — slight growth during training |
| readout (`hidden2rgb`) | 0.067 |
| modul | 1.861 — essentially unchanged from init (~1.84) |

Implied global Lipschitz bound (sine_only, freq=30): `σ(W_rgb) * Π(30·σ(W_ℓ)) ≈ 2.7×10^5`.

### What this tells us about L calibration

The vanilla-learned SIREN has an **extremely non-uniform** σ distribution: the first sine layer sits at ~5.0 (dominated by init; input coords cover [0,27]² so the model needs that wide Wx spread to feed the sine), the hidden sine layers at ~0.1, and `modul` at ~1.9 (init). Therefore:

- At `L=30`, `apply_to=sine_only`: sine.0 gets σ-cap 1.0 but has σ=4.98 → the **first layer is the only layer that gets any penalty**. Other sines are at 0.1 << 1.0 (inactive). This is a meaningful but surgical constraint.
- At `L=150`, `apply_to=sine_only`: sine.0 cap becomes 5.0, still right at its natural σ → near-zero pressure. Basically "vanilla +ε".
- At `L=3`, `apply_to=sine_only`: sine.0 cap becomes 0.1, a ~50× compression on the first layer; hidden sines at σ=0.1 right at their cap too. Expect reconstruction degradation.
- If you want to constrain ALL the sine layers, use `apply_to=sine_only` with a small L but *accept that it will shrink the first sine layer out of its natural regime*.

This informs the follow-up sweep.

## 11. Files touched

| File | Status | Summary |
|---|---|---|
| `SIREN_Vista/trainer.py` | modified | resume flag, variant wiring, penalty hook, slugged savedir, richer checkpoint payload, MSE/penalty split in tqdm log (§9), `--log-sigmas-every` (§10) |
| `SIREN_Vista/makeset.py` | modified | variant flags + pre-load `variants.build(...)` |
| `SIREN_Vista/variants/__init__.py` | new | registry + dispatch |
| `SIREN_Vista/variants/vanilla.py` | new | default no-op variant |
| `SIREN_Vista/variants/soft_lipschitz.py` | new + later modified | first real robustness variant; cap→L semantic update (§7) |
| `SIREN_Vista/diagnostics.py` | new | `layer_sigmas` + `format_sigmas_one_liner` (§10) |
| `SIREN_Vista/evaluate_reconstruction.py` | new | reconstruction MSE/PSNR/SSIM eval with multi-iter snapshots (§8) |
| `SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh` | modified | Step 4/4 eval call; `EVAL_ITERS`, `EVAL_MAX_SAMPLES`, `LOG_SIGMAS_EVERY` knobs |
| `SIREN_Vista/SIREN.py`, `utils.py`, dataloaders, `train_classifier.py`, `attacks/` | unchanged | — |

## 12. Adding a new variant (for future chats)

1. Create `SIREN_Vista/variants/<new_variant>.py`.
2. Inside it, decorate a class with `@register("<name>")` and implement `add_args`, `build`, `penalty`, `slug`.
3. Import the new module at the bottom of `variants/__init__.py` so registration fires on `import variants`.
4. Add an entry to `_extract_variant_args`'s `prefix_map` in `variants/__init__.py` so saved checkpoints contain a compact record of the variant's flags.

No changes to `trainer.py` or `makeset.py` are needed unless the variant reshapes `state_dict` keys in a way that requires custom load logic.

## 13. CIFAR-10 support and Fourier-SIREN INR backbone

**Files.** `SIREN_Vista/{SIREN.py,dataloader.py,trainer.py,makeset.py,train_classifier.py,evaluate_reconstruction.py}` and `SIREN_Vista/scripts/{run_soft_cifar10.sh,run_vanilla_cifar10_big.sh,run_fourier_cifar10.sh}`.

Added CIFAR-10 as a 2D RGB dataset path:

- `dataloader.py` now has `get_cifar10_loader(...)`.
- `SIREN.py`'s 2D SIREN path supports `out_features=3`, so CIFAR targets are `(32*32, 3)` rather than grayscale `(H*W, 1)`.
- `trainer.py`, `makeset.py`, and `evaluate_reconstruction.py` accept `dataset=cifar10`.
- `evaluate_reconstruction.py` computes RGB MSE/PSNR over all channels and RGB SSIM by averaging per-channel SSIM.

Added an INR-backbone choice separate from the `variants/` regularization system:

```bash
--inr-type siren
--inr-type fourier_siren
--fourier-num-freqs 64
--fourier-sigma 10.0
--fourier-include-input
```

New classes in `SIREN.py`:

- `FourierFeatureEncoding`: fixed random Gaussian Fourier basis with registered buffer `fourier.B`, mapping `(x,y)` to `[sin(2πBx), cos(2πBx)]`, optionally concatenating raw coordinates.
- `ModulatedFourierSIREN`: same modulation mechanism as `ModulatedSIREN` (`phi -> modul -> per-layer SIREN shifts`), but the coordinate path is `coords -> FourierFeatureEncoding -> SineAffine stack -> hidden2rgb`.

Checkpoint metadata now records INR provenance:

```python
model_args = {
    "dataset": ...,
    "hidden_dim": ...,
    "mod_dim": ...,
    "depth": ...,
    "height": ...,
    "width": ...,
    "out_features": ...,
    "inr_type": "siren" or "fourier_siren",
    "fourier_num_freqs": ...,
    "fourier_sigma": ...,
    "fourier_include_input": ...,
}
```

Important loading fix:

- `makeset.py` now loads the checkpoint first, reads `checkpoint["model_args"]`, and rebuilds the exact architecture before `load_state_dict(...)`.
- `evaluate_reconstruction.py` does the same and overrides CLI defaults from checkpoint metadata.
- This fixes the previously observed failure where `run_vanilla_cifar10_big.sh` trained a `hidden_dim=512`, `mod_dim=1024` checkpoint but `makeset.py` rebuilt the default `256/512` model and hit a large `size mismatch` error.

Script status:

- `scripts/run_soft_cifar10.sh` was reduced to a vanilla-only CIFAR-10 pipeline for the first small CIFAR baseline.
- `scripts/run_vanilla_cifar10_big.sh` now defaults to `SKIP_STEP1=1`, so it skips retraining if the big checkpoint already exists, passes architecture flags to `makeset.py` / eval, and saves functaset pickles under the run slug via `--functaset-stem`.
- `scripts/run_fourier_cifar10.sh` is the first Fourier-SIREN CIFAR experiment: `hidden_dim=256`, `mod_dim=512`, `depth=10`, `fourier_num_freqs=64`, `fourier_sigma=10.0`, `epochs=5`, `makeset_iters=20`.

Verification performed:

- Python compile passed for modified files.
- `bash -n scripts/run_fourier_cifar10.sh` passed after normalizing CRLF line endings.
- Smoke tests confirmed:
  - `ModulatedFourierSIREN(height=32,width=32,out_features=3)` outputs `(1024, 3)`.
  - `fourier.B` is saved in the state dict.
  - `evaluate_reconstruction.batched_forward(...)` works for Fourier-SIREN and RGB output.
  - `makeset.py` / `evaluate_reconstruction.py` can rebuild a Fourier-SIREN from checkpoint `model_args`.
