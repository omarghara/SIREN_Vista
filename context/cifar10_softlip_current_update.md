# CIFAR-10 Softlip Current Status Update

Updated: 2026-05-30

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

| model | classifier checkpoint | best test top-1 |
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
- samples: first `200` CIFAR-10 test images
- PGD steps: `200`
- PGD LR: `0.01`
- inner phi fit steps during attack: `5`
- inner phi LR: `0.01`
- clean/final phi clipping: disabled
- epsilons: `{1, 2, 4, 6, 8}/255`

| model | eps (/255) | n | clean acc | robust acc | robust given clean |
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

## Current Interpretation

The corrected attack now gives comparable clean accuracy. On this first-200
subset, softlip is slightly better than vanilla both clean and robust:

- clean: `82.0%` softlip vs `79.0%` vanilla
- eps `1/255`: `57.5%` robust vs `54.0%`
- eps `2/255`: `31.5%` robust vs `28.0%`
- eps `4/255`: `6.5%` robust vs `4.0%`

The signal is positive but still small. By eps `6/255`, both models are almost
fully broken, and by eps `8/255`, both have `0%` robust accuracy on this
subset.

Next step: rerun the corrected no-clip protocol on a larger sample count,
starting with `n=1000`, and then add attack-strength checks such as PGD LR
sweeps, random restarts, and more inner fitting steps.
