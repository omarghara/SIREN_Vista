# CIFAR-10 Spatial Functa Robustness Status

Updated: 2026-06-03

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
- samples: first `1000` CIFAR-10 test examples for the current estimate;
  first-`200` rows are retained only for comparison with earlier quick runs
- classifier: newly trained CNN over the matching inner-5 spatial phi

Completed patched no-clip matched inner-5 PGD-200 sweep:

| model | eps (/255) | n | clean acc | robust acc | robust \| clean |
|---|---:|---:|---:|---:|---:|
| vanilla e512 inner5 | 1 | 1000 | 0.761 | 0.527 | 0.693 |
| softlip tiered e12 inner5 | 1 | 1000 | 0.763 | 0.534 | 0.700 |
| vanilla e512 inner5 | 1 | 200 | 0.790 | 0.540 | 0.684 |
| softlip tiered e12 inner5 | 1 | 200 | 0.820 | 0.575 | 0.701 |
| vanilla e512 inner5 | 2 | 1000 | 0.761 | 0.312 | 0.410 |
| softlip tiered e12 inner5 | 2 | 1000 | 0.763 | 0.319 | 0.418 |
| vanilla e512 inner5 | 2 | 200 | 0.790 | 0.280 | 0.354 |
| softlip tiered e12 inner5 | 2 | 200 | 0.820 | 0.315 | 0.384 |
| vanilla e512 inner5 | 4 | 1000 | 0.761 | 0.044 | 0.0578 |
| softlip tiered e12 inner5 | 4 | 1000 | 0.763 | 0.064 | 0.0839 |
| vanilla e512 inner5 | 4 | 200 | 0.790 | 0.040 | 0.0506 |
| softlip tiered e12 inner5 | 4 | 200 | 0.820 | 0.065 | 0.0793 |
| vanilla e512 inner5 | 6 | 1000 | 0.761 | 0.003 | 0.0039 |
| softlip tiered e12 inner5 | 6 | 1000 | 0.763 | 0.005 | 0.0066 |
| vanilla e512 inner5 | 6 | 200 | 0.790 | 0.000 | 0.000 |
| softlip tiered e12 inner5 | 6 | 200 | 0.820 | 0.005 | 0.0061 |
| vanilla e512 inner5 | 8 | 1000 | 0.761 | 0.003 | 0.0039 |
| softlip tiered e12 inner5 | 8 | 1000 | 0.763 | 0.001 | 0.0013 |
| vanilla e512 inner5 | 8 | 200 | 0.790 | 0.000 | 0.000 |
| softlip tiered e12 inner5 | 8 | 200 | 0.820 | 0.000 | 0.000 |

On `n=1000`, softlip tiered has nearly matched clean accuracy (`0.763` vs
`0.761`) and a small robust-accuracy edge at eps `1/255`, `2/255`, `4/255`,
and `6/255`. The largest difference is at eps `4/255`: softlip has `64/1000`
robust samples versus vanilla `44/1000`. At eps `8/255`, both are essentially
broken and vanilla is slightly higher (`3/1000` vs `1/1000`), which is within
tiny-count noise.

The earlier first-200 subset showed a larger clean gap (`0.820` vs `0.790`),
but the `n=1000` sweep is the better current estimate.

Delta, softlip minus vanilla, `n=1000`:

| eps (/255) | clean delta | robust delta | robust \| clean delta |
|---:|---:|---:|---:|
| 1 | +0.002 | +0.007 | +0.0074 |
| 2 | +0.002 | +0.007 | +0.0081 |
| 4 | +0.002 | +0.020 | +0.0261 |
| 6 | +0.002 | +0.002 | +0.0026 |
| 8 | +0.002 | -0.002 | -0.0026 |

### Warm-start regularizer sweep on vanilla e512

After the matched inner-5 vanilla/softlip comparison, several new CIFAR
backbones were meta-trained by warm-starting from the vanilla e512 checkpoint
and adding late-layer spectral or orthogonality penalties. Each completed
backbone then used the same inner-5 pipeline:

```text
make functaset with 5 phi steps -> train CNN classifier -> PGD-200 with 5 phi steps
```

Artifacts:

- warm-start training launcher:
  `scripts/run_cifar10_spatial_warmstart_regularizer.sh`
- generic checkpoint evaluation launcher:
  `scripts/run_cifar10_spatial_inner5_checkpoint.sh`
- result root:
  `runs/cifar10_spatial_inner5_warmstart_models`
- detailed experiment notes:
  `context/cifar10_warmstart_regularizer_experiments.md`

Classifier validation summary:

| model | best val top-1 | status |
|---|---:|---|
| `warm_readout_cap10_lam1` | 76.86% | complete |
| `warm_prereadout_cap10_lam1` | 76.75% | complete |
| `warm_prereadout_counter1_lam1e-2` | 76.76% | complete |
| `warm_readout_cap50_lam1` | 76.42% | complete |
| `warm_orth_lam1e-3` | 76.27% | complete |
| `warm_readout_counter1_lam1e-2` | 70.20% | classifier interrupted after epoch 10 |
| `warm_readout_cap90_lam1` | missing | makeset interrupted before classifier |

PGD-200 results, `n=200`:

| model | eps1 robust | eps2 robust | eps4 robust | eps6 robust | eps8 robust | clean acc |
|---|---:|---:|---:|---:|---:|---:|
| vanilla e512 inner5 | 0.540 | 0.280 | 0.040 | 0.000 | 0.000 | 0.790 |
| softlip tiered e12 inner5 | 0.575 | 0.315 | 0.065 | 0.005 | 0.000 | 0.820 |
| `warm_readout_cap10_lam1` | 0.530 | 0.320 | 0.040 | 0.000 | 0.000 | 0.790 |
| `warm_prereadout_cap10_lam1` | 0.545 | 0.345 | 0.020 | 0.000 | 0.000 | 0.815 |
| `warm_readout_cap50_lam1` | 0.500 | 0.310 | 0.040 | 0.005 | 0.005 | 0.830 |
| `warm_prereadout_counter1_lam1e-2` | 0.535 | 0.280 | 0.040 | 0.000 | 0.000 | 0.840 |
| `warm_orth_lam1e-3` | 0.490 | 0.260 | 0.010 | 0.005 | 0.000 | 0.770 |
| `warm_readout_counter1_lam1e-2` | 0.515 | 0.395 | 0.070 | 0.025 | missing | 0.720 |

Interpretation:

- The completed 76%-class warm-start variants do not clearly beat the
  softlip tiered baseline. They mostly match or trail it at eps `1/255` and
  eps `4/255`.
- `warm_prereadout_cap10_lam1` is the best completed 76%-class warm-start
  result at eps `2/255` (`0.345` robust), but it falls to `0.020` by eps
  `4/255`.
- `warm_readout_counter1_lam1e-2` has the best robust accuracy at eps `2`,
  `4`, and `6`, and the best robust-given-clean ratios, but its classifier
  was interrupted early and only reached `70.20%` validation top-1, so it is
  a promising stress test rather than an apples-to-apples improvement.
- `warm_readout_cap90_lam1` still needs a clean rerun of makeset and
  classifier before it can be compared.

### Softlip-warmstart cap sweep

After the vanilla-warmstart sweep, four new runs were warm-started from the
softlip tiered e12 checkpoint itself and capped either the RGB readout or the
final sine layer relative to the softlip checkpoint's own singular values.

Artifacts:

- launcher:
  `scripts/run_cifar10_spatial_softlip_warmstart_cap_pipeline.sh`
- backbone roots:
  `model_cifar10/cifar10_spatial_warmsoftlip_*_lam1.0_e5`
- evaluation root:
  `runs/cifar10_spatial_inner5_softlip_warmstart_caps`

Common protocol:

- warm-start/checkpoint reference: softlip tiered e12
- train 5 meta-learning epochs
- `spectral_cap`, `reference_scale`, `lambda = 1.0`
- make inner-5 functaset
- train the same CNN classifier
- PGD-200 with attack `--mod-steps 5`
- eps `{1,2,4,6}/255`, `n=200`

Loss and cap convergence:

| model | best total loss | final MSE | final penalty | saved sigma / cap | cap status |
|---|---:|---:|---:|---:|---|
| `warmsoftlip_readout_cap50_lam1.0_e5` | `0.002912` | `0.002311` | `0.000601` | `1.32x` | did not fully reach cap |
| `warmsoftlip_readout_cap10_lam1.0_e5` | `0.008562` | `0.003182` | `0.005528` | `5.94x` | did not reach cap |
| `warmsoftlip_prereadout_cap50_lam1.0_e5` | `0.001914` | `0.001863` | `0.000051` | `1.02x` | almost reached cap |
| `warmsoftlip_prereadout_cap10_lam1.0_e5` | `0.025236` | `0.001853` | `0.023382` | `4.20x` | did not reach cap |

The exact SVD audit showed that the 50% final-sine cap nearly landed. The 50%
readout cap moved in the intended direction but remained above cap. Both 10%
caps were still strongly violating their requested cap after 5 epochs.

Classifier and PGD-200 results:

| model | best val top-1 | clean | eps1 robust | eps2 robust | eps4 robust | eps6 robust |
|---|---:|---:|---:|---:|---:|---:|
| softlip tiered e12 inner5 baseline | 75.73% | 0.820 | 0.575 | 0.315 | 0.065 | 0.005 |
| `warmsoftlip_readout_cap50_lam1.0_e5` | 75.91% | 0.790 | 0.575 | 0.340 | 0.040 | 0.015 |
| `warmsoftlip_readout_cap10_lam1.0_e5` | 75.93% | 0.795 | 0.515 | 0.335 | 0.065 | 0.010 |
| `warmsoftlip_prereadout_cap50_lam1.0_e5` | 76.04% | 0.805 | 0.530 | 0.345 | 0.045 | 0.000 |
| `warmsoftlip_prereadout_cap10_lam1.0_e5` | 76.95% | 0.815 | 0.535 | 0.320 | 0.025 | 0.005 |

Interpretation:

- These runs do not clearly improve on the existing softlip tiered baseline.
- `warmsoftlip_readout_cap50_lam1.0_e5` is the best mixed result: it matches
  softlip at eps `1/255`, improves eps `2/255` and eps `6/255`, but loses at
  eps `4/255` and has lower clean accuracy.
- `warmsoftlip_prereadout_cap50_lam1.0_e5` has the best eps `2/255` robust
  accuracy among this mini-sweep, but it is worse at eps `1/255`, `4/255`,
  and `6/255`.
- The strict 10% caps should be interpreted as under-converged pressure
  experiments, not as successful 10% spectral caps.

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
- In the patched no-clip matched inner-5 PGD sweep, the completed `n=1000`
  run gives softlip tiered nearly equal clean accuracy to vanilla and a small
  robust/conditional-robust advantage at eps `1/255`, `2/255`, and `4/255`.
- The older cap90 run showed a weak-positive robustness signal at eps
  `2/255` and above, but that run used `10` attack fitting steps and is no
  longer the cleanest comparison.

What is not enough yet:

- In the patched matched inner-5 `n=1000` rerun, eps `6/255` leaves only
  `3/1000` robust vanilla samples and `5/1000` robust softlip samples; eps
  `8/255` leaves `3/1000` vanilla and `1/1000` softlip.
- The previous softlip tiered clean accuracy drop inside PGD (`61%`) was
  caused by attack-time clean/final refit gradient clipping. With clipping
  disabled, clean accuracy on the first 200 samples is `79%` vanilla and
  `82%` softlip tiered.
- Softlip cap90 has much lower clean accuracy on the attacked subset
  (`62.5%` vs `75.5%`), so the comparison is not clean-accuracy matched.
- Absolute robust accuracy is still very low by eps `4/255`.
- The matched inner-5 softlip advantage remains small even at `n=1000`, so it
  is a real signal to investigate rather than a final robustness claim.
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

1. Treat the vanilla e512 vs softlip tiered e12 inner-5 PGD-200, `n=1000`,
   eps `{1,2,4,6,8}/255` sweep as the current pinned CIFAR baseline.

2. Rerun the promising or incomplete warm-start leads cleanly:
   - `warm_readout_counter1_lam1e-2` with a full 40-epoch classifier and
     eps `8/255`
   - `warm_readout_cap90_lam1` from makeset through classifier and PGD
   - optionally milder counter targets to keep clean classifier accuracy near
     the 76% baseline

3. Treat the new softlip-warmstart cap sweep as a negative/diagnostic result:
   it reduces spectral amplification diagnostics, but it does not dominate
   softlip tiered on PGD. If revisited, the 10% caps need more epochs or a
   different schedule before they can be evaluated as actually-capped models.

4. Add attack-strength checks for the corrected protocol:
   - PGD LR sweep
   - random restarts
   - inner mod steps `10` vs `20`
   - BPDA or transfer-style checks if PGD behavior looks suspicious

5. Run a CIFAR cap/reference-scale sweep:
   - reference-scaled caps, e.g. `0.95`, `0.90`, `0.85`
   - optionally include a mild modulation cap
   - keep reconstruction and clean classifier accuracy as constraints

6. Add CIFAR modulation-stability diagnostics:
   - `||phi(x + delta) - phi(x)|| / ||delta||`
   - classifier-logit change per input perturbation
   - classifier-logit change per phi perturbation

7. Track the old clipped-run numbers only as a debugging artifact, not as
   CIFAR evidence.

## Current Working Thesis Statement

For CIFAR-10, Spatial Functa makes the parameter-space pipeline viable:
good reconstruction and roughly 70% clean classification are now possible.
The patched no-clip matched inner-5 rerun fixed both the fitting-budget
mismatch and the attack-time clean-refit gradient-clipping bug. Softlip
tiered has a small robustness edge at eps `1/255` to `4/255`; on the
completed `n=1000` sweep it has nearly equal clean accuracy (`76.3%` vs
`76.1%`) and the clearest edge at eps `4/255` (`6.4%` vs `4.4%` robust).
Both models still collapse by eps `6/255` to `8/255`. The first warm-start
cap/orthogonal sweep did not produce a clean dominant replacement:
the 76%-class cap variants mostly trail softlip tiered, while the aggressive
readout-counter variant looks more robust only with a lower, interrupted
classifier. The follow-up softlip-warmstart cap sweep also does not clearly
dominate softlip tiered, and the strict 10% caps did not converge within 5
epochs. Any publishable result needs larger-sample confirmation, a pinned
adaptive attack protocol, and modulation-stability diagnostics.
