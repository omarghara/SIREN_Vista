# Supervisor sync — thesis progress update

Brief for the next supervisor meeting. Paths below are relative to `/home/omarg/` unless noted.

---

## 1. Recap of the thesis goal

Master's thesis on **adversarial robustness of parameter-space classifiers built on SIREN implicit neural representations**. Pipeline:

```
signal x  →  inner INR fit  →  modulation z*(x)  →  classifier g(z*(x))  →  label
```

Upstream paper: *Adversarial Attacks in Weight-Space Classifiers* (Shor et al., arXiv:2502.20314): apparent robustness is largely **gradient obfuscation** from the inner fitting loop. Thesis aim: distinguish accidental from **structured** robustness under **adaptive** attacks.

---

## 2. What exists in the codebase

### 2.1 Research catalogue — `ideas/robustness_ideas.md`

Twenty-four candidate defenses in ten categories (Lipschitz / spectral / modulation stability / amortized fitting / adversarial training in \(z\)-space, etc.).

### 2.2 Engineering infrastructure (`SIREN_Vista/`)

- **Resume + provenance** (`trainer.py --resume`; checkpoints carry variant metadata).
- **Variants plug-in** (`variants/`): `vanilla`, `soft_lipschitz`; four-method pattern (`add_args`, `build`, `penalty`, `slug`).
- **Spectral diagnostics** (`diagnostics.py`; `--log-sigmas-every N` during training).
- **Reconstruction metrics** (`evaluate_reconstruction.py`): MSE / PSNR / SSIM vs inner-loop iters (batched).
- **Full-PGD** (`attacks/full_pgd.py`): `--max-samples`, `--output-json`, unconditional + **conditional** robust accuracy.
- **Pipelines:**
  - `scripts/run_soft_lipschitz_mnist.sh` — train softlip → functaset → classifier → recon eval → PGD.
  - `scripts/run_pgd_plan.sh` — softlip **vs vanilla** PGD main run + \(\varepsilon\) sweep + merged `runs/pgd_plan_summary.{md,json}`.
  - `scripts/run_pgd_single_model.sh` — same PGD recipe for **one** slug only (no vanilla stage).
- **Diagnostics notebook** (`SIREN_Vista/notebooks/model_diagnostics.ipynb`): weight \(\sigma\) tables, pre/post-sine activations, coordinate-scale audit, \(|\nabla_x f|\), recon curves, perturbation sensitivity / optional heavy PGD + 2D slice.

Details: `context/CHANGES.md`, `context/project_state.md`.

### 2.3 Defense implemented — soft Lipschitz

Training-time penalty \(\lambda \sum_\ell \max(0, \sigma(W_\ell) - c_\ell)^2\) with caps from a single budget \(L\) (sine: \(L/\omega_0\); readout: \(L\); `modul` when `apply_to=all`: \(L/\omega_0\)); optional **skip `sine.0`**.

---

## 3. Experimental results (current story)

### 3.1 Backbone reconstruction (`context/report.md`)

Same backbone shape (depth 10, hidden 256, modul 512, \(\omega_0=30\)). Compared run: **vanilla** (repo 6-epoch SIREN) vs **softlip** `L=30`, \(\lambda=1\), `apply_to=all`, `skip_first` (slug `softlip_L30_lam1e+00_all_skip0`).

At inner iters **= 5** (classifier operating point):

| | vanilla | softlip | \(\Delta\) |
|--|--------:|--------:|----------:|
| PSNR (dB) | 20.3 | 16.2 | **−4.1** |
| SSIM | 0.72 | 0.53 | −0.18 |

Gap ~4 dB PSNR is **flat** through iters 5–200 → softlip has a **lower fidelity ceiling**, not just slower fitting.

### 3.2 Full-PGD — **matched** vanilla e40 vs softlip (`context/PGD_ROBUSTNESS_REPORT.md`)

**Vanilla baseline** is no longer the old 6-epoch / default-path stack. It is the fully trained bundle:

- **Slug:** `vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256`
- **SIREN:** `SIREN_Vista/model_mnist/vanilla_e40_.../modSiren.pth`
- **Classifier:** README-style — 40 epochs, lr 0.01, width 512, mod 512, dropout 0.2, depth 3, batch 256 (`runs/vanilla_e40_.../`).

**Softlip compared:** `softlip_L30_lam1e+00_all_skip0` (same as recon / earlier plan).

**Protocol:** `attacks/full_pgd.py`, 100 outer PGD steps, outer lr 0.01, 10 inner mod steps @ 0.01, L∞ \(\varepsilon/255\). Samples: \(n=500\) at \(\varepsilon=16\); \(n=200\) at \(\varepsilon \in \{8,32,64\}\).

**Headline table (robust accuracy — who wins depends on \(\varepsilon\)):**

| \(\varepsilon\) (/255) | \(n\) | softlip robust | vanilla e40 robust | Winner (robust) |
|--:|--:|--:|--:|:--|
| 8 | 200 | 0.915 | **0.950** | vanilla |
| 16 | 500 | 0.678 | **0.712** | vanilla |
| 32 | 200 | **0.340** | 0.215 | softlip |
| 64 | 200 | **0.265** | 0.030 | softlip |

**Clean accuracy** on these evaluation slices: vanilla ~**99%**, softlip ~**97.5%** — not matched-clean; vanilla is stronger on standard accuracy here.

**Takeaway:** With a **strong vanilla**, the story is **regime-dependent**: vanilla is **better** under Full-PGD at **small–medium** \(\varepsilon\) (8, 16 here); softlip retains a **large** advantage at **high** \(\varepsilon\) (e.g. 64/255: **26.5%** vs **3%** robust). Any write-up must name the vanilla checkpoint and report both regimes.

*(Earlier **softlip-only** PGD logs in `context/attack_currenct_results.md` — e.g. ~81% clean / 63% robust at \(\varepsilon=16\) on a growing subset — used a weaker reporting context; the **matched** numbers above supersede that for thesis claims.)*

### 3.3 Training-time \(\sigma\) log (still relevant)

At **init** with `apply_to=all`, `L=30`: `sine.1–9` and readout sit **below** cap; **`modul`** sits **above** \(L/\omega_0\) → penalty bites **modul** first. **End-of-training** \(\sigma\) audit on the saved `.pth` is still the right check that this remained the active bottleneck.

### 3.4 CIFAR-10 scaling and Fourier-SIREN pivot

We added CIFAR-10 support to the functa pipeline (`32x32` RGB output) and observed that the first small vanilla CIFAR SIREN baseline is weak relative to MNIST:

- Reconstruction around **PSNR@200 ≈ 18.8 dB**.
- Downstream classifier around **47% top-1**.

This suggests the immediate CIFAR bottleneck is representation fidelity, not only robustness. A larger vanilla run (`scripts/run_vanilla_cifar10_big.sh`, `hidden_dim=512`, `mod_dim=1024`, `makeset_iters=50`) was prepared; a bug in `makeset.py` was found and fixed where checkpoint metadata was ignored and the default `256/512` model was rebuilt.

New controlled representation experiment: **Fourier-SIREN**.

- New INR flag: `--inr-type fourier_siren`.
- Coordinate path: `(x,y) -> fixed Gaussian Fourier features -> SIREN`.
- Modulation path unchanged: `phi -> modul -> per-layer shifts`; classifier still trains on `phi`.
- Checkpoint metadata records `inr_type`, Fourier frequency count, sigma, include-input flag, image shape, output channels.
- `makeset.py` and `evaluate_reconstruction.py` now rebuild from checkpoint `model_args`, preventing architecture mismatch.

First script to run: `SIREN_Vista/scripts/run_fourier_cifar10.sh`:

```text
hidden_dim=256, mod_dim=512, depth=10
fourier_num_freqs=64, fourier_sigma=10.0, include_input=False
epochs=5, makeset_iters=20
```

Success criterion for this phase: improve CIFAR reconstruction over vanilla (`PSNR@200 > 20–21 dB` would be encouraging) and then check whether better reconstruction also improves modulation-space classifier accuracy beyond ~47%.

---

## 4. Interpretation

**Structured trade-off:** Softlip pays ~4 dB PSNR in the backbone; under PGD it **does not** uniformly beat a **properly trained** vanilla — at \(\varepsilon=8\) and **16** vanilla e40 wins on robust accuracy **and** clean accuracy in this protocol.

**Where softlip still matters for the thesis:** At **large** \(\varepsilon\), softlip keeps **non-trivial** robust accuracy while vanilla **collapses** (~3% at 64/255 vs ~26.5%). That supports a nuanced claim: **Lipschitz-style control can help under strong attacks** even when it loses at moderate budgets to a strong unconstrained baseline.

**Not gradient-only obfuscation at a glance:** Attack success is material; \(\varepsilon\) sweep is not flat in a masking-like way — but stronger claims still want sweeps + (optional) transfer / black-box checks per `cursor_context.md`.

**CIFAR interpretation:** Fourier-SIREN is not a thesis pivot; it is a controlled change of the INR representation while preserving the same parameter-space classification pipeline. The question remains how INR choice and regularization affect `phi` quality/stability and downstream robustness.

**Coordinate normalization fix:** Both SIREN models now use pixel-centre normalised coordinates in `[0,1]` (helpers `make_normalized_pixel_grid`), matching the From-Data-to-Functa convention. For Fourier features this is critical — raw integer pixel indices would cause random Fourier phases to be much too large. All models must be retrained from scratch after this change for a fair comparison against any previous checkpoints.

---

## 5. Open questions for supervisor

1. **Thesis framing:** Lead with **regime-dependent** PGD results (strong vanilla at low \(\varepsilon\), softlip at high \(\varepsilon\)) vs. trying to sell softlip as uniformly “more robust.”
2. **Further soft-Lipschitz work:** Tune **\(L\), \(\lambda\)** (including runs like **`L=0.5`** with `apply_to=all` where caps are very tight), or pivot toward **explicit `modul`-only** regularization / **hard spectral norm** / **modulation-stability** from the ideas list?
3. **Evaluation depth:** Is **Full-PGD + \(\varepsilon\) sweep** enough for the defense chapter, or do we need **AutoAttack**, **Square/NES**, or **transfer** runs for an exam committee?
4. **Datasets:** Stay **MNIST-only** until the defense story is frozen, or add Fashion-MNIST / ModelNet10 for one chosen variant?
5. **CIFAR representation:** If Fourier-SIREN improves CIFAR reconstruction/classification, should the robustness chapter include it as the scalable INR baseline before revisiting soft-Lipschitz / hardcap regularization?

---

## 6. Suggested next steps

1. **Write the PGD section** of the thesis from `PGD_ROBUSTNESS_REPORT.md` (figures: robust vs \(\varepsilon\), table with conditional metrics). Cite **`vanilla_e40_...`** explicitly.
2. **Offline \(\sigma\) audit** on `softlip_L30_lam1e+00_all_skip0` (and any new slug) via `diagnostics.layer_sigmas` / notebook — confirm which layers hit caps **after** training.
3. **Optional ablations:** new softlip checkpoints (e.g. **`L=0.5`** pipeline already in `run_soft_lipschitz_mnist.sh`); compare PGD via `run_pgd_single_model.sh` or extend the comparison script.
4. **Diagnostics notebook** on vanilla e40 + chosen softlip for the “where is sharpness / first layer scale” narrative (see `to_do/chat.md`).
5. **If time:** one **transfer** or **black-box** sanity check; or **matched-clean** training (harder — only if reviewer angle requires it).
6. **Run Fourier-SIREN CIFAR pilot** via `scripts/run_fourier_cifar10.sh`; compare `reconstruction_eval.json` and classifier accuracy against `vanilla_cifar10`.

---

## 7. Artifact index

| Kind | Location |
|------|----------|
| PGD comparison write-up | `context/PGD_ROBUSTNESS_REPORT.md` |
| Reconstruction report | `context/report.md` |
| Earlier softlip-only PGD notes | `context/attack_currenct_results.md` |
| Changelog / handoff | `context/CHANGES.md`, `context/project_state.md` |
| Vanilla e40 artifacts | `SIREN_Vista/model_mnist/vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256/`, `SIREN_Vista/runs/vanilla_e40_.../` |
| Merged PGD summaries (when run) | `SIREN_Vista/runs/pgd_plan_summary.{md,json}` |
| Per-model PGD JSON/log | `SIREN_Vista/runs/<slug>/pgd_plan/` |
| Notebooks | `SIREN_Vista/notebooks/model_diagnostics.ipynb` |
| CIFAR diagnostics notebook | `SIREN_Vista/notebooks/model_reconstruction_tsne_perturbation_cifar10.ipynb` |
| Scripts | `SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh`, `run_pgd_plan.sh`, `run_pgd_single_model.sh`, `run_vanilla_cifar10_big.sh`, `run_fourier_cifar10.sh`, **`run_spatial_cifar10.sh`** |
| Ideas | `ideas/robustness_ideas.md` |

---

## 8. One-line summary for the meeting

“We now have a **proper vanilla e40 baseline** and a **matched Full-PGD \(\varepsilon\) sweep**: vanilla **wins** robust accuracy at \(\varepsilon=8\) and **16**, softlip **wins** at **32** and **64** — so the thesis story is **nuanced**, not ‘softlip beats vanilla everywhere’. On CIFAR-10, vanilla SIREN reconstruction/classification is weak (~18.8 dB PSNR@200, ~47% top-1), so I added a controlled **Fourier-SIREN INR** path to test whether better coordinate features improve reconstruction and `phi` quality before revisiting robustness.”

---

## 9. Latest development — Spatial Functa (May 2026)

Implemented **Spatial Functa** (Dupont et al. 2022 Sec. 4 + App. C.1) as an optional,
backwards-compatible extension of the existing functa pipeline.

### Core idea

Replace the global modulation vector `phi ∈ ℝ^{mod_dim}` with a **spatial latent grid**
`phi ∈ ℝ^{s×s×c}`. Each pixel gets its latent code from the nearest grid cell (1-NN),
feeds it through a per-pixel `Linear(c, L*hidden)` to produce per-pixel shifts, and the
INR backbone runs with those local shifts. Paper Table 4 CIFAR-10 config: `s=8, c=16`
→ `phi_numel=1024`.

### Files changed

| File | What changed |
|------|--------------|
| `SIREN.py` | New `SpatialModulatedINR`; `init_phi` / `phi_shape` / `phi_numel` / `is_spatial` on all existing classes; per-pixel shifts in all three affine types |
| `trainer.py` | Spatial CLI flags; `_build_2d_model` dispatch; `fit()` via `model.init_phi()` |
| `makeset.py` | Same; functaset entries carry `phi_shape` + `is_spatial` |
| `train_classifier.py` | `_flatten_modulations` helper; `--classifier-type` flag |
| `evaluate_reconstruction.py` | Spatial + FINER + FourierLSA branches; slow per-image inner loop |
| `scripts/run_spatial_cifar10.sh` | **New** end-to-end pipeline (paper `d=6, h=256, 8×8×16`) |

All spatial behavior gated behind `--spatial-modulation` (default off). Existing MNIST /
CIFAR vanilla / softlip pipelines are **unchanged** — confirmed by smoke tests across all
phases A → J.

### Expected sanity output (run at model-build time via `[build]` prints)

```
[build] SpatialModulatedINR  base_inr_type=siren  is_spatial=True  phi_shape=(8, 8, 16)  phi_numel=1024
```

### Verification (added May 2026)

A formal audit of `SpatialModulatedINR` in `SIREN.py` confirmed the implementation is
**correct** against the paper's spatial latent-to-modulation pipeline:

| Check | Result |
|-------|--------|
| φ shape `(s, s, c)` | ✓ |
| 1-NN cell assignment | ✓ pixels `(0-3, *)` → cell-row 0; `(4-7, *)` → cell-row 1 |
| Local patch coords | ✓ `0.125, 0.375, 0.625, 0.875` for 4-px patches |
| `Linear(z_cell)` ≡ `Conv2d(1×1)` | ✓ max diff = 0.0 (proved + commented in SIREN.py) |
| Gradient flow to φ `(8, 8, 16)` | ✓ |

New file: `scripts/test_spatial_functa.py` (5 tests, all pass).

### How to run the paper config

```bash
PRESET=paper bash ~/SIREN_Vista/scripts/run_spatial_functa_cifar10_subset.sh
```

Config: SIREN/256/depth-6/ω₀=10/batch-128/ext_lr=3e-5, 511 epochs ≈ 200k updates,
makeset 3 SGD inner steps, local coords, nearest-neighbor, 8×8×16 grid.
