# CIFAR-10 Spatial Functa Robustness Status

Updated: 2026-05-30

## Active Goal

The current project focus is CIFAR-10. The goal is to show that a
Lipschitz-regularized INR backbone can make a parameter-space classifier
more robust to PGD attacks on the input image.

The pipeline remains:

```text
CIFAR-10 image x
   -> fit spatial modulation grid phi(x)
   -> classify phi(x)
   -> attack x while differentiating through phi fitting
```

The classifier does not see pixels directly. Robustness means:

```text
small pixel perturbation
   -> smaller fitted-phi change
   -> classifier prediction remains stable
```

## Why Spatial Functa

Standard/global SIREN-style Functa was not enough for CIFAR-10. Earlier
attempts with vanilla SIREN, Fourier variants, FINER, and LSA-style
models struggled to combine good reconstruction with a useful downstream
classifier.

The important scaling pivot was the Spatial Functa paper:

```text
Spatial Functa: Scaling Functa to ImageNet Classification and Generation
https://arxiv.org/abs/2302.03130
```

That paper explicitly identifies scaling problems for ordinary Functa on
CIFAR-10-like data and proposes spatially arranged latent representations.
After implementing this direction, this repo finally reached good CIFAR-10
reconstruction quality and semi-good classifier accuracy around 70%, with
the current matched inner-5 CNN classifiers reaching about 76%.

## Current Implementation

Main implementation points:

- `SIREN.py::SpatialModulatedINR`
- spatial latent grid: `8 x 8 x 16`
- interpolation: nearest / 1-NN
- coordinates: local coordinates inside each latent cell
- modulation type: shift-only
- backbone: mostly SIREN for the current robust-vs-vanilla comparison
- CIFAR image output: RGB, 32 x 32
- downstream classifier: CNN over spatial phi, not flat MLP
- attack: `attacks/full_pgd_cifar10_spatial.py`

The spatial model maps each pixel to a latent cell, converts that cell code
to per-pixel layer shifts, reconstructs RGB, and the classifier operates on
the learned spatial phi grid.

## Key CIFAR Checkpoints

### Vanilla Spatial SIREN

Slug:

```text
functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000
```

Backbone:

```text
model_cifar10/<slug>/modSiren.pth
```

Important notes:

- checkpoint summary records best backbone loss around `0.0032`
- original flat MLP classifier only reached about `49%`
- CNN classifier is the relevant classifier for attack logs
- PGD logs report CNN classifier checkpoint accuracy `71.74%`
- on the first 200 PGD samples, clean accuracy is `75.5%`

### Soft-Lipschitz cap90 Spatial SIREN

Slug:

```text
functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cap90_lam1e-02_sine_and_readout_train50000_test10000
```

Backbone:

```text
model_cifar10/<slug>/modSiren.pth
```

Classifier:

```text
runs/<slug>/cifar10_cnn_classifier/best_classifier.pth
```

Important notes:

- CNN classifier checkpoint summary reports best top-1 `66.90%`
- on the first 200 PGD samples, clean accuracy is `62.5%`
- this clean-accuracy gap versus vanilla must be handled carefully

### Soft-Lipschitz tiered Spatial SIREN

Slug:

```text
functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cifar_spatial_tiered_lam1e-02_sine_and_readout_train50000_test10000
```

Important notes:

- full CNN classifier checkpoint summary reports best top-1 `66.94%`
- classifier sweep found better settings around `69.69%` validation top-1
- retrained sweep checkpoint reports test top-1 `70.98%`
- quick reconstruction is strong:
  - 3 fit steps: PSNR `27.38`, SSIM `0.881`
  - 200 fit steps: PSNR `29.08`, SSIM `0.914`
  - 1000 fit steps: PSNR `30.13`, SSIM `0.929`
- PGD quick outputs for this slug show suspiciously low clean accuracy
  (`31%` to `36%`) and should not be used as a final robustness claim until
  the exact classifier checkpoint and normalization path are verified.

## Current PGD Results

Important correction as of 2026-05-30:

`attacks/full_pgd_cifar10_spatial.py` was patched so the clean/final phi
refit defaults to `--clean-grad-clip 0.0`. This matches `makeset.py`, which
uses plain SGD on phi without gradient clipping. The previous matched
inner-5 PGD rerun used gradient clipping inherited from the original
Parameter-Space-Attack-Suite `full_pgd.py`; that clipping made the 5-step
attack-time refit underfit, especially for softlip tiered, and produced a
false clean-accuracy gap (`0.760` vanilla vs `0.610` softlip). Treat those
clipped PGD numbers as invalid for the current CIFAR comparison.

### Matched inner-5 rerun: vanilla e512 vs softlip tiered e12

This is now the most relevant CIFAR attack protocol because it removes the
previous train/attack fitting-budget mismatch. The functasets were refit with
`5` inner phi steps, the CNN classifiers were trained on those new inner-5
functasets, and PGD also uses `--mod-steps 5`.

Artifacts:

- scripts:
  - `scripts/run_cifar10_spatial_inner5_vanilla.sh`
  - `scripts/run_cifar10_spatial_inner5_softlip.sh`
- experiment root:
  `runs/cifar10_spatial_inner5_make5_clfbest_v1`
- vanilla functaset:
  `runs/cifar10_spatial_inner5_make5_clfbest_v1/vanilla_e512/functaset/vanilla_e512_inner5_train_all50000.pkl`
- softlip tiered functaset:
  `runs/cifar10_spatial_inner5_make5_clfbest_v1/softlip_tiered_e12/functaset/softlip_tiered_e12_inner5_train_all50000.pkl`

Classifier logs:

| model | classifier | best logged top-1 |
|---|---|---:|
| vanilla e512 inner5 | CNN, best sweep params | 76.27% |
| softlip tiered e12 inner5 | CNN, best sweep params | 75.73% |

Attack protocol:

- script: `attacks/full_pgd_cifar10_spatial.py`
- PGD steps: `200`
- PGD LR: `0.01`
- inner phi steps: `5`
- inner phi LR: `0.01`
- clean/final phi grad clipping: disabled, matching `makeset.py`
- samples: first `200` CIFAR-10 test examples
- classifier: newly trained CNN over the matching inner-5 spatial phi

Completed patched no-clip matched inner-5 PGD-200 sweep:

| model | eps (/255) | n | clean acc | robust acc | robust \| clean |
|---|---:|---:|---:|---:|---:|
| vanilla e512 inner5 | 1 | 200 | 0.790 | 0.540 | 0.684 |
| softlip tiered e12 inner5 | 1 | 200 | 0.820 | 0.575 | 0.701 |
| vanilla e512 inner5 | 2 | 200 | 0.790 | 0.280 | 0.354 |
| softlip tiered e12 inner5 | 2 | 200 | 0.820 | 0.315 | 0.384 |
| vanilla e512 inner5 | 4 | 200 | 0.790 | 0.040 | 0.0506 |
| softlip tiered e12 inner5 | 4 | 200 | 0.820 | 0.065 | 0.0793 |
| vanilla e512 inner5 | 6 | 200 | 0.790 | 0.000 | 0.000 |
| softlip tiered e12 inner5 | 6 | 200 | 0.820 | 0.005 | 0.0061 |
| vanilla e512 inner5 | 8 | 200 | 0.790 | 0.000 | 0.000 |
| softlip tiered e12 inner5 | 8 | 200 | 0.820 | 0.000 | 0.000 |

Softlip tiered has slightly higher clean accuracy on this first-200 subset
and slightly higher unconditional and conditional robust accuracy at eps
`1/255`, `2/255`, and `4/255`. At eps `6/255`, vanilla has `0/200` robust
samples and softlip has `1/200`; at eps `8/255`, both are fully broken on
this 200-image subset.

Delta, softlip minus vanilla:

| eps (/255) | clean delta | robust delta | robust \| clean delta |
|---:|---:|---:|---:|
| 1 | +0.030 | +0.035 | +0.0177 |
| 2 | +0.030 | +0.035 | +0.0297 |
| 4 | +0.030 | +0.025 | +0.0287 |
| 6 | +0.030 | +0.005 | +0.0061 |
| 8 | +0.030 | +0.000 | +0.000 |

### Earlier PGD-200 sweep with attack inner phi steps = 10

These older results are still useful historical context, but they should not
be mixed with the matched inner-5 rerun above because the classifiers/functa
creation used a smaller fitting budget than the attack.

Attack protocol for those saved results:

- script: `attacks/full_pgd_cifar10_spatial.py`
- PGD steps: `200`
- PGD LR: `0.01`
- inner phi steps: `10`
- inner phi LR: `0.01`
- samples: first `200` CIFAR-10 test examples
- classifier: CNN over spatial phi

### Small-Epsilon Sweep

| model | eps (/255) | clean acc | robust acc | robust \| clean |
|---|---:|---:|---:|---:|
| vanilla spatial SIREN | 1 | 0.755 | 0.400 | 0.530 |
| softlip cap90 spatial SIREN | 1 | 0.625 | 0.215 | 0.336 |
| vanilla spatial SIREN | 2 | 0.755 | 0.105 | 0.139 |
| softlip cap90 spatial SIREN | 2 | 0.625 | 0.140 | 0.216 |
| vanilla spatial SIREN | 4 | 0.755 | 0.005 | 0.0066 |
| softlip cap90 spatial SIREN | 4 | 0.625 | 0.035 | 0.056 |
| vanilla spatial SIREN | 6 | 0.755 | 0.005 | 0.0066 |
| softlip cap90 spatial SIREN | 6 | 0.625 | 0.015 | 0.024 |
| vanilla spatial SIREN | 8 | 0.755 | 0.005 | 0.0066 |
| softlip cap90 spatial SIREN | 8 | 0.625 | 0.015 | 0.024 |

### Larger Epsilon Sweep

| model | eps (/255) | clean acc | robust acc | robust \| clean |
|---|---:|---:|---:|---:|
| vanilla spatial SIREN | 16 | 0.755 | 0.005 | 0.0066 |
| softlip cap90 spatial SIREN | 16 | 0.625 | 0.015 | 0.024 |
| vanilla spatial SIREN | 32 | 0.755 | 0.005 | 0.0066 |
| softlip cap90 spatial SIREN | 32 | 0.625 | 0.015 | 0.024 |
| vanilla spatial SIREN | 64 | 0.755 | 0.005 | 0.0066 |
| softlip cap90 spatial SIREN | 64 | 0.625 | 0.015 | 0.024 |

Ignore the `eps16_n1.json` artifact in the softlip cap90 folder; it is a
one-sample smoke test, not an experiment.

## Interpretation

The current CIFAR evidence is mixed and not thesis-ready yet.

What looks promising:

- Spatial Functa solved the main reconstruction bottleneck for CIFAR-10.
- CNN classifiers over spatial phi can reach roughly `67%` to `72%`.
- The matched inner-5 classifiers themselves are healthy: `76.27%` logged
  top-1 for vanilla and `75.73%` for softlip tiered.
- In the patched no-clip matched inner-5 PGD sweep, softlip tiered has a
  small clean, robust, and conditional-robust advantage over vanilla at eps
  `1/255`, `2/255`, and `4/255`.
- The older cap90 run showed a weak-positive robustness signal at eps
  `2/255` and above, but that run used `10` attack fitting steps and is no
  longer the cleanest comparison.

What is not enough yet:

- In the patched matched inner-5 rerun, eps `6/255` leaves `0/200` robust
  vanilla samples and only `1/200` robust softlip samples; eps `8/255`
  robust accuracy is `0.0` for both models on 200 images.
- The previous softlip tiered clean accuracy drop inside PGD (`61%`) was
  caused by attack-time clean/final refit gradient clipping. With clipping
  disabled, clean accuracy on the first 200 samples is `79%` vanilla and
  `82%` softlip tiered.
- Softlip cap90 has much lower clean accuracy on the attacked subset
  (`62.5%` vs `75.5%`), so the comparison is not clean-accuracy matched.
- Absolute robust accuracy is still very low by eps `4/255`.
- The matched inner-5 softlip advantage is small and based on only 200
  samples, so it is a signal to investigate rather than a final robustness
  claim.
- Some saved summaries mix old classifiers, MLP classifiers, tuned CNNs, and
  smoke-test artifacts. Always check the exact classifier path in the PGD log.

## Important Gotchas

1. `runs/pgd_cifar10_spatial_cnn_summary.md` is not the current complete
   comparison. It currently summarizes an older vanilla-only run with lower
   clean accuracy.

2. The vanilla `run_summary.md` points to a flat MLP classifier with about
   `49%` top-1. The PGD-200 runs use a CNN classifier whose log reports
   `71.74%`. Use the CNN for robust comparisons.

3. The old softlip tiered PGD quick results with very low clean accuracy
   (`31%` to `36%`) and the first matched inner-5 clipped rerun (`61%`
   clean) should not be used for claims. The current patched attack disables
   clean/final refit clipping and restores the expected first-200 clean
   accuracy.

4. For the current matched inner-5 rerun, use only the new classifier
   checkpoints under
   `runs/cifar10_spatial_inner5_make5_clfbest_v1/*/cifar10_cnn_classifier_best_sweep_inner5/`.
   These are trained on the new inner-5 functasets and should be paired with
   PGD `--mod-steps 5` and default `--clean-grad-clip 0.0`.

5. `variants/soft_lipschitz.py` has drifted into an experiment harness. If no
   reference checkpoint is supplied, the active code uses hardcoded
   vanilla-relative caps (`first95_rest80`) rather than the generic
   `L/freq` cap described by the CLI help. Future CIFAR experiments should
   make the cap scheme explicit in the slug and checkpoint metadata.

## Immediate Next Steps

1. Pin one CIFAR attack protocol:
   - vanilla spatial SIREN e512
   - softlip tiered e12
   - same CNN classifier family
   - same first-N test subset
   - PGD-200
   - eps `{1,2,4,6,8}/255`
   - same fitting budget for functaset creation, classifier evaluation, and
     attack (`5` inner phi steps for the current rerun)

2. Scale the patched no-clip matched protocol to larger sample counts
   (`n=1000` first) to check whether the small softlip edge persists.

3. Add attack-strength checks for the corrected protocol:
   - PGD LR sweep
   - random restarts
   - inner mod steps `10` vs `20`
   - BPDA or transfer-style checks if PGD behavior looks suspicious

4. Run a CIFAR cap/reference-scale sweep:
   - reference-scaled caps, e.g. `0.95`, `0.90`, `0.85`
   - optionally include a mild modulation cap
   - keep reconstruction and clean classifier accuracy as constraints

5. Add CIFAR modulation-stability diagnostics:
   - `||phi(x + delta) - phi(x)|| / ||delta||`
   - classifier-logit change per input perturbation
   - classifier-logit change per phi perturbation

6. Track the old clipped-run numbers only as a debugging artifact, not as
   CIFAR evidence.

## Current Working Thesis Statement

For CIFAR-10, Spatial Functa makes the parameter-space pipeline viable:
good reconstruction and roughly 70% clean classification are now possible.
The patched no-clip matched inner-5 rerun fixed both the fitting-budget
mismatch and the attack-time clean-refit gradient-clipping bug. Softlip
tiered has a small robustness edge at eps `1/255` to `4/255` while also
having slightly higher clean accuracy on the first 200 samples (`82%` vs
`79%`), but both models collapse by eps `6/255` to `8/255`. Any publishable
result needs larger-sample confirmation, a pinned adaptive attack protocol,
and modulation-stability diagnostics.
