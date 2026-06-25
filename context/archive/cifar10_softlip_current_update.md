# CIFAR-10 Softlip Current Status Update

Updated: 2026-06-03

## Goal

We are testing whether a Lipschitz-regularized Spatial Functa SIREN gives a
more robust CIFAR-10 parameter-space classifier under PGD attacks.

The pipeline is:

```text
CIFAR-10 image
  -> fit spatial phi grid with the INR
  -> classify phi with a CNN
  -> attack the image while differentiating through phi fitting
```

## What Softlip Means Here

The current softlip model is:

```text
functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cifar_spatial_tiered_lam1e-02_sine_and_readout_train50000_test10000
```

It is a Spatial Functa SIREN:

- image size: CIFAR-10 RGB, `32 x 32`
- spatial phi shape: `8 x 8 x 16`
- total phi size: `1024`
- SIREN hidden width: `256`
- SIREN depth: `6` sine layers
- sine frequency: `10.0`
- spatial interpolation: nearest cell lookup
- coordinates: local coordinates inside each spatial cell
- modulation: shift-only, generated from each local phi cell

The softlip constraint is a training-time spectral-norm penalty. It does not
replace the weights with spectrally normalized weights. Instead, during INR
training we add:

```text
lambda * sum_l max(0, sigma(W_l) - cap_l)^2
```

where `sigma(W_l)` is the largest singular value of a layer weight matrix,
estimated with power iteration, and `lambda = 0.01`.

For this CIFAR run we used the checkpoint preset:

```text
soft_lip_cap_preset = cifar_spatial_tiered
```

This was a vanilla-relative tiered spectral cap. The intended pattern was:

- first sine layer: about `90%` of the vanilla spectral norm
- next two sine layers: about `85%` of the vanilla spectral norm
- later sine layers: about `75%` of the vanilla spectral norm
- modulation map `phi -> shifts`: not capped in this experiment

The saved checkpoint confirms this pattern when comparing the learned
softlip singular values to the vanilla checkpoint:

| layer | vanilla sigma | softlip sigma | softlip / vanilla |
|---|---:|---:|---:|
| `sine.0` | `4.9763` | `4.4793` | `0.900` |
| `sine.1` | `0.4234` | `0.3658` | `0.864` |
| `sine.2` | `0.5490` | `0.4678` | `0.852` |
| `sine.3` | `0.4600` | `0.3495` | `0.760` |
| `sine.4` | `0.4839` | `0.3661` | `0.756` |
| `sine.5` | `0.6128` | `0.4613` | `0.753` |
| `hidden2rgb` readout | `0.1824` | `0.1513` | `0.829` |

So this is not a global `L/freq` cap in the current CIFAR experiment. It is a
vanilla-relative, layer-wise cap: mild on the first coordinate-input sine
layer, medium on the next hidden layers, and stronger on the later hidden
layers. The current `variants/soft_lipschitz.py` helper has since drifted to
a different hardcoded fallback, so the checkpoint metadata and measured
checkpoint sigmas are the safer source for this run.

## Classifier

The classifier does not see pixels. It classifies the fitted spatial phi grid.

For the current matched comparison, we created new functasets by fitting each
CIFAR-10 image with exactly `5` inner SGD steps:

```text
image -> fit phi for 5 SGD steps -> save phi -> train classifier on phi
```

The classifier is `SpatialPhiCNN`:

- input phi shape before preprocessing: `(8, 8, 16)`
- classifier input after preprocessing: `(16, 8, 8)`
- architecture:
  - `Conv2d(16 -> 256) + BatchNorm + ReLU`
  - `Conv2d(256 -> 256) + BatchNorm + ReLU`
  - `Conv2d(256 -> 512) + BatchNorm + ReLU`
  - global average pooling
  - dropout `0.1`
  - linear layer to 10 CIFAR classes
- training:
  - learning rate `0.003`
  - batch size `256`
  - epochs `40`
  - phi normalization enabled using training-set `phi_mean` and `phi_std`

Classifier checkpoints:

| model | classifier checkpoint | best logged top-1 |
|---|---|---:|
| vanilla e512 inner5 | `runs/cifar10_spatial_inner5_make5_clfbest_v1/vanilla_e512/cifar10_cnn_classifier_best_sweep_inner5/best_classifier.pth` | `76.27%` |
| softlip tiered e12 inner5 | `runs/cifar10_spatial_inner5_make5_clfbest_v1/softlip_tiered_e12/cifar10_cnn_classifier_best_sweep_inner5/best_classifier.pth` | `75.73%` |

## Important PGD Fix

We found a mismatch in the attack evaluation.

`makeset.py` fits phi with plain SGD and no gradient clipping. The CIFAR PGD
attack had inherited clean/final phi gradient clipping from the original
`Parameter-Space-Attack-Suite/attacks/full_pgd.py`. With only 5 inner steps,
that clipping made the attack-time phi underfit the image, especially for
softlip.

After patching `attacks/full_pgd_cifar10_spatial.py`, the default is now:

```text
--clean-grad-clip 0.0
```

This matches the functaset creation path. The old clipped PGD numbers should
not be used for the current CIFAR comparison.

## Current Corrected PGD Results

Protocol:

- attack: `attacks/full_pgd_cifar10_spatial.py`
- samples: first `1000` CIFAR-10 test images for the current estimate
- PGD steps: `200`
- PGD LR: `0.01`
- inner phi fit steps during attack: `5`
- inner phi LR: `0.01`
- clean/final phi clipping: disabled
- epsilons: `{1, 2, 4, 6, 8}/255`

| model | eps (/255) | n | clean acc | robust acc | robust given clean |
|---|---:|---:|---:|---:|---:|
| vanilla e512 inner5 | 1 | 1000 | 0.761 | 0.527 | 0.693 |
| softlip tiered e12 inner5 | 1 | 1000 | 0.763 | 0.534 | 0.700 |
| vanilla e512 inner5 | 2 | 1000 | 0.761 | 0.312 | 0.410 |
| softlip tiered e12 inner5 | 2 | 1000 | 0.763 | 0.319 | 0.418 |
| vanilla e512 inner5 | 4 | 1000 | 0.761 | 0.044 | 0.0578 |
| softlip tiered e12 inner5 | 4 | 1000 | 0.763 | 0.064 | 0.0839 |
| vanilla e512 inner5 | 6 | 1000 | 0.761 | 0.003 | 0.0039 |
| softlip tiered e12 inner5 | 6 | 1000 | 0.763 | 0.005 | 0.0066 |
| vanilla e512 inner5 | 8 | 1000 | 0.761 | 0.003 | 0.0039 |
| softlip tiered e12 inner5 | 8 | 1000 | 0.763 | 0.001 | 0.0013 |

## Warm-Started Follow-Up Experiments

After the matched vanilla-vs-softlip comparison, we tried several new
regularizer experiments by warm-starting from the trained vanilla e512 Spatial
SIREN checkpoint instead of training from scratch.

The common idea was:

```text
start from vanilla e512 SIREN
  -> continue meta-training with an extra constraint penalty
  -> build a new inner-5 functaset
  -> train the same CNN classifier on phi
  -> run matched inner-5 PGD-200
```

The warm-start trainer optimizes:

```text
total loss = reconstruction MSE + regularizer penalty
```

The verification notebook is:

```text
notebooks/cifar10_warmstart_regularizer_verification.ipynb
```

It computes exact SVD-based singular values, cap violations, orthogonality
penalties, and layerwise amplification bounds from the saved checkpoints. These
diagnostics are also saved under:

```text
runs/diagnostics/cifar10_warmstart_regularizer_verification
```

### What Each Experiment Tried

| experiment | intended constraint |
|---|---|
| `warm_orth_lam1e-3` | Make sine layers and readout close to orthogonal using `mean((G - I)^2)`, with `G = W^T W` for square/tall layers and `W W^T` where needed. |
| `warm_readout_cap90_lam1` | Cap only the RGB readout spectral norm to `90%` of the vanilla readout sigma. |
| `warm_readout_cap50_lam1` | Cap only the RGB readout spectral norm to `50%` of the vanilla readout sigma. |
| `warm_readout_cap10_lam1` | Cap only the RGB readout spectral norm to `10%` of the vanilla readout sigma. |
| `warm_prereadout_cap10_lam1` | Cap the final sine layer before the readout to `10%` of the vanilla final-sine sigma. |
| `warm_readout_counter1_lam1e-2` | Compute the vanilla product amplification and try to counter it only through a very small readout cap. |
| `warm_prereadout_counter1_lam1e-2` | Compute the vanilla product amplification and try to counter it only through a very small final-sine cap. |

### Did The New Constraints Actually Converge?

The table below uses exact SVD diagnostics from the verification notebook.
`checkpoint loss` is the saved trainer loss for the checkpoint. For cap models,
`weighted penalty` is `lambda * sum(max(0, sigma - cap)^2)`. For the orthogonal
model, it is `lambda * sum(mean((G - I)^2))`.

| model | epoch | checkpoint loss | target check | weighted penalty | amplification bound | reached intended cap/constraint? |
|---|---:|---:|---|---:|---:|---|
| softlip tiered e12 | 57 | 0.001905 | max cap violation `0.005932` | `9.91e-07` | 6846 | Mostly yes. Layer sigmas match the intended tiered vanilla-relative caps closely. |
| `warm_orth_lam1e-3` | 19 | 0.086606 | max layer orth penalty `82.267` | `8.26e-02` | 22138 | No. Hidden square layers are small, but `sine.0` dominates the orthogonality error. |
| `warm_readout_cap90_lam1` | 1 | 0.001221 | readout `0.164538` vs cap `0.164199` | `1.15e-07` | 25249 | Basically yes, but the checkpoint only reached epoch 1 and was not a completed evaluated model. |
| `warm_readout_cap50_lam1` | 19 | 0.001720 | readout `0.101600` vs cap `0.091222` | `1.08e-04` | 15790 | Not fully. It is close-ish, but still above the cap. |
| `warm_readout_cap10_lam1` | 19 | 0.006318 | readout `0.082407` vs cap `0.018244` | `4.12e-03` | 13284 | No. The cap was too strong; readout stayed about `4.52x` above the cap. |
| `warm_prereadout_cap10_lam1` | 19 | 0.001976 | final sine `0.085869` vs cap `0.061275` | `6.05e-04` | 6747 | Partially. It reduced the final sine strongly, but did not reach the 10% cap. |
| `warm_readout_counter1_lam1e-2` | 10 | 0.001591 | readout `0.136925` vs cap `6.34e-06` | `1.87e-04` | 20767 | No. The counter cap is unrealistically tiny; actual sigma is about `21601x` the cap. |
| `warm_prereadout_counter1_lam1e-2` | 19 | 0.001500 | final sine `0.184689` vs cap `2.13e-05` | `3.41e-04` | 10895 | No. The counter cap is also far too tiny; actual sigma is about `8675x` the cap. |

Main diagnostic conclusion:

- Softlip tiered is still the cleanest example where the intended spectral
  constraint mostly happened and reconstruction/classification remained usable.
- `warm_prereadout_cap10_lam1` reduced the amplification bound to about the
  same level as softlip tiered (`6747` vs `6846`), but it did not improve PGD
  robustness.
- Capping only the readout lowers the product bound, but strong readout caps
  did not fully converge and did not produce a clear robustness gain.
- The counter-amplification caps are mathematically too aggressive if applied
  only to one late layer.
- The orthogonality experiment did not make the full SIREN orthogonal because
  the first sine layer remains the dominant problem.

### Warm-Start Classifier Results

All completed warm-start pipelines used the same inner-5 functaset/classifier
setup as the vanilla/softlip comparison.

| model | best logged classifier top-1 | status |
|---|---:|---|
| `warm_readout_cap10_lam1` | 76.86% | complete |
| `warm_prereadout_cap10_lam1` | 76.75% | complete |
| `warm_prereadout_counter1_lam1e-2` | 76.76% | complete |
| `warm_readout_cap50_lam1` | 76.42% | complete |
| `warm_orth_lam1e-3` | 76.27% | complete |
| `warm_readout_counter1_lam1e-2` | 70.20% | classifier interrupted after epoch 10 |
| `warm_readout_cap90_lam1` | missing | makeset/classifier pipeline incomplete |

### Warm-Start PGD Results

These warm-start PGD results are on `n=200`, not `n=1000`, so they should be
treated as quick comparison evidence. The vanilla and softlip rows below are
the matching first-200 rows from the corrected inner-5 protocol.

| model | clean | eps1 robust | eps2 robust | eps4 robust | eps6 robust | eps8 robust |
|---|---:|---:|---:|---:|---:|---:|
| vanilla e512 inner5 | 0.790 | 0.540 | 0.280 | 0.040 | 0.000 | 0.000 |
| softlip tiered e12 inner5 | 0.820 | 0.575 | 0.315 | 0.065 | 0.005 | 0.000 |
| `warm_readout_cap10_lam1` | 0.790 | 0.530 | 0.320 | 0.040 | 0.000 | 0.000 |
| `warm_prereadout_cap10_lam1` | 0.815 | 0.545 | 0.345 | 0.020 | 0.000 | 0.000 |
| `warm_readout_cap50_lam1` | 0.830 | 0.500 | 0.310 | 0.040 | 0.005 | 0.005 |
| `warm_prereadout_counter1_lam1e-2` | 0.840 | 0.535 | 0.280 | 0.040 | 0.000 | 0.000 |
| `warm_orth_lam1e-3` | 0.770 | 0.490 | 0.260 | 0.010 | 0.005 | 0.000 |
| `warm_readout_counter1_lam1e-2` | 0.720 | 0.515 | 0.395 | 0.070 | 0.025 | missing |

Comparison to what we already had:

- None of the completed 76%-class warm-start variants clearly beats the
  existing softlip tiered model.
- `warm_prereadout_cap10_lam1` is the best completed warm-start model at eps
  `2/255` (`0.345` vs softlip `0.315`), but it is worse at eps `4/255`
  (`0.020` vs softlip `0.065`) and collapses at eps `6/255`.
- `warm_readout_cap10_lam1` slightly beats softlip at eps `2/255`
  (`0.320` vs `0.315`) but otherwise does not improve the result.
- `warm_readout_cap50_lam1` has high clean accuracy and keeps `1/200` robust
  samples at eps `8/255`, but it is lower than softlip at eps `1/255`,
  `2/255`, and `4/255`.
- `warm_orth_lam1e-3` is worse than both vanilla and softlip almost everywhere.
- `warm_readout_counter1_lam1e-2` looks most robust at eps `2/255`,
  `4/255`, and `6/255`, but it is not a fair win because its classifier was
  interrupted and clean accuracy is only `0.720`.

## Softlip-Warmstarted 5-Epoch Cap Sweep

We then ran four additional experiments that warm-start from the existing
softlip tiered checkpoint instead of the vanilla checkpoint.

Common setup:

- warm-start checkpoint: softlip tiered e12
- cap reference checkpoint: the same softlip tiered e12 checkpoint
- variant: `spectral_cap`
- cap mode: `reference_scale`
- penalty weight: `lambda = 1.0`
- training length: `5` meta-learning epochs
- downstream pipeline: make inner-5 functaset, train the same CNN classifier,
  run PGD-200 on the first `200` CIFAR-10 test images
- PGD epsilons: `{1, 2, 4, 6}/255`

The launch script for this sweep is:

```text
scripts/run_cifar10_spatial_softlip_warmstart_cap_pipeline.sh
```

The two reference singular values from the softlip tiered checkpoint were:

| target layer | softlip reference sigma |
|---|---:|
| final sine layer `siren.net.5.affine.weight` | `0.461347` |
| RGB readout `siren.hidden2rgb.weight` | `0.151287` |

### Loss Convergence

The trainer objective was:

```text
total = reconstruction MSE + lambda * max(0, sigma(W_target) - cap)^2
```

| model | target | epoch 0 total / MSE / pen | final total / MSE / pen | best epoch | best total |
|---|---|---:|---:|---:|---:|
| `warmsoftlip_readout_cap50_lam1.0_e5` | readout, 50% cap | `0.004915 / 0.002954 / 0.001961` | `0.002912 / 0.002311 / 0.000601` | 4 | `0.002912` |
| `warmsoftlip_readout_cap10_lam1.0_e5` | readout, 10% cap | `0.013154 / 0.003538 / 0.009617` | `0.008710 / 0.003182 / 0.005528` | 3 | `0.008562` |
| `warmsoftlip_prereadout_cap50_lam1.0_e5` | final sine, 50% cap | `0.020835 / 0.002206 / 0.018629` | `0.001914 / 0.001863 / 0.000051` | 4 | `0.001914` |
| `warmsoftlip_prereadout_cap10_lam1.0_e5` | final sine, 10% cap | `0.100623 / 0.002212 / 0.098411` | `0.025236 / 0.001853 / 0.023382` | 4 | `0.025236` |

The 50% final-sine cap is the only one that nearly converged to zero penalty
within 5 epochs. The readout 50% cap reduced the penalty but did not reach
zero. Both 10% caps were still heavily penalty-dominated at the end.

### Did The Caps Actually Land?

This table uses exact SVD on the saved checkpoints, not the training-time
power-iteration estimates.

| model | cap | saved sigma | sigma / cap | weighted penalty | reached cap? |
|---|---:|---:|---:|---:|---|
| `warmsoftlip_readout_cap50_lam1.0_e5` | `0.075644` | `0.099906` | `1.32x` | `5.89e-04` | No, but moved toward it. |
| `warmsoftlip_readout_cap10_lam1.0_e5` | `0.015129` | `0.089827` | `5.94x` | `5.58e-03` | No. |
| `warmsoftlip_prereadout_cap50_lam1.0_e5` | `0.230674` | `0.236185` | `1.02x` | `3.04e-05` | Almost. |
| `warmsoftlip_prereadout_cap10_lam1.0_e5` | `0.046135` | `0.193868` | `4.20x` | `2.18e-02` | No. |

The product-style spectral amplification bound still decreased relative to
the softlip reference (`6846`):

| model | product bound after training |
|---|---:|
| `warmsoftlip_readout_cap50_lam1.0_e5` | `4732` |
| `warmsoftlip_readout_cap10_lam1.0_e5` | `4374` |
| `warmsoftlip_prereadout_cap50_lam1.0_e5` | `4420` |
| `warmsoftlip_prereadout_cap10_lam1.0_e5` | `4081` |

So the interventions did reduce this simple amplification diagnostic, but
the strongest requested caps did not actually land within 5 epochs.

### Classifier And PGD Results

All rows below use the matched inner-5 classifier/attack protocol and
`n=200` PGD-200.

| model | best classifier top-1 | clean | eps1 robust | eps2 robust | eps4 robust | eps6 robust |
|---|---:|---:|---:|---:|---:|---:|
| softlip tiered e12 inner5 baseline | `75.73%` | `0.820` | `0.575` | `0.315` | `0.065` | `0.005` |
| `warmsoftlip_readout_cap50_lam1.0_e5` | `75.91%` | `0.790` | `0.575` | `0.340` | `0.040` | `0.015` |
| `warmsoftlip_readout_cap10_lam1.0_e5` | `75.93%` | `0.795` | `0.515` | `0.335` | `0.065` | `0.010` |
| `warmsoftlip_prereadout_cap50_lam1.0_e5` | `76.04%` | `0.805` | `0.530` | `0.345` | `0.045` | `0.000` |
| `warmsoftlip_prereadout_cap10_lam1.0_e5` | `76.95%` | `0.815` | `0.535` | `0.320` | `0.025` | `0.005` |

Interpretation:

- None of the softlip-warmstarted cap models clearly beats the original
  softlip tiered baseline.
- `warmsoftlip_readout_cap50_lam1.0_e5` is the most interesting row: it
  matches softlip at eps `1/255`, improves eps `2/255` and eps `6/255`, but
  loses clean accuracy and is worse at eps `4/255`.
- `warmsoftlip_readout_cap10_lam1.0_e5` matches softlip at eps `4/255` and
  improves eps `2/255`, but the requested 10% readout cap did not converge.
- `warmsoftlip_prereadout_cap50_lam1.0_e5` has the best eps `2/255` robust
  accuracy among these four (`0.345`), but it trails softlip at eps `1/255`
  and eps `4/255`.
- `warmsoftlip_prereadout_cap10_lam1.0_e5` has the best classifier top-1 and
  high clean accuracy, but the strict final-sine cap did not converge and its
  eps `4/255` robustness is poor.

## Current Interpretation

The corrected attack now gives comparable clean accuracy. On the first 1000
test images, softlip is nearly clean-matched to vanilla and slightly more
robust at eps `1/255`, `2/255`, `4/255`, and `6/255`:

- clean: `76.3%` softlip vs `76.1%` vanilla
- eps `1/255`: `53.4%` robust vs `52.7%`
- eps `2/255`: `31.9%` robust vs `31.2%`
- eps `4/255`: `6.4%` robust vs `4.4%`

The signal is positive but still small. By eps `6/255`, both models are
almost fully broken (`5/1000` robust for softlip, `3/1000` for vanilla).
By eps `8/255`, both are essentially broken; vanilla has `3/1000` robust
samples and softlip has `1/1000`.

The new vanilla-warmstart and softlip-warmstart experiments did not replace
the softlip tiered result. They were still useful diagnostically: they show
that reducing a simple product-of-spectral-norm amplification bound is
measurable, but that alone does not guarantee better PGD robustness for the
full pipeline `image -> fitted phi -> classifier`.

Current working hypothesis:

- constraining only the SIREN readout or final sine layer is probably not
  enough;
- the product spectral bound is too loose to explain robustness by itself;
- the sensitive part may be the fitted representation map `x -> phi*(x)`, the
  downstream CNN classifier over phi, or the full composed map
  `x -> phi*(x) -> logits`.

Next steps:

1. Add attack-strength checks such as PGD LR sweeps, random restarts, more PGD
   steps, and more inner phi-fitting steps.
2. Rerun only the genuinely promising/incomplete warm-start lead,
   `warm_readout_counter1_lam1e-2`, with a full classifier if we want to test
   whether its robust-looking numbers survive matched clean accuracy.
3. Move beyond only capping SIREN weights and test diagnostics/regularizers on
   `phi`, `logits`, `d logits / d phi`, and the stability of the inner
   fitting map `x -> phi*(x)`.
