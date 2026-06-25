# 01 — Project and pipeline

## What this project is
A master's thesis on **parameter-space (weight/modulation-space) classifiers built
on implicit neural representations (INRs)**, and their **adversarial robustness**.

Instead of classifying pixels, we classify the *fitted representation* of an image.
The active dataset is **CIFAR-10** using a **Spatial Functa**-style INR
(`SIREN.py::SpatialModulatedINR`, latent grid `8 x 8 x 16`).

Upstream reference repo (the paper we build on):
`Parameter-Space-Attack-Suite` — "Adversarial Robustness in Parameter-Space
Classifiers". Key papers: SIREN (2006.09661), Functa (2201.12204),
Spatial Functa (2302.03130), and the weight-space attack paper
(OpenReview eOLybAlili).

## The bilevel pipeline (this is the whole point)
```text
image x
  -> (inner loop) fit a spatial modulation grid  phi(x) = argmin_phi || f_theta(phi) - x ||^2
  -> (classifier) CNN over spatial phi -> predicted label
```
An attack perturbs the **pixels** x, but the label is decided **after** the inner
fit. So robustness depends on the *composition*:
```text
x -> phi(x) -> g(phi(x))
```
Two sensitivities matter:
- how much the fitted `phi` moves when `x` is perturbed (encoder/fitting stability), and
- how much the classifier `g` reacts to changes in `phi`.

We track the per-input amplification:
```text
A(x, delta) = || phi(x+delta) - phi(x) ||_2 / || delta ||_2
```
Lower `A` = more stable representation under perturbation.

## Critical warning (do not forget)
This is a **nested / bilevel** system, so naive white-box PGD can *underestimate*
vulnerability via **gradient masking**. A defense that merely makes gradients hard
to compute looks robust but is not. Always corroborate with stronger/adaptive
attacks before believing any robustness number.

## Repo map (the files that matter)
| Path | Role |
|---|---|
| `SIREN.py` | INR architecture (`SpatialModulatedINR`), sine layers, modulation, readout. First place to change architecture. |
| `trainer.py` | Meta-trains the SIREN backbone. Hosts soft-Lipschitz penalty hooks and the **hard SVD projection** post-step hook. |
| `spectral_projection.py` | Hard SVD projection module (clamp singular values after each optimizer step). Targets incl. `readout`, `pre_readout`, `all_sine_readout`, `modul`, `modul_readout`. |
| `makeset.py` | Builds the **functaset** (fit phi for every image; train/val/test/all50000 `.pkl`). |
| `train_classifier.py` | Trains the CNN classifier over spatial phi. |
| `attacks/full_pgd_cifar10_spatial.py` | PGD attack through the inner fit (pixel-space L-inf). |
| `variants/` | Regularizer variant definitions (caps, counter-amplification, orthogonality, etc.). |
| `dataloader.py` | CIFAR-10 / data loading. |
| `evaluate_reconstruction.py` | Reconstruction/PSNR helpers. |
| `scripts/` | Pipeline launchers + eval scripts (see `05_how_to_run.md`). |
| `notebooks/` | Diagnostics, incl. `cifar10_latest_robustness_layer_analysis.ipynb` (layer amplification). |

## Eval/diagnostic scripts (session-built)
- `scripts/reconstruct_compare.py` — reconstruct an image with a checkpoint, report PSNR.
- `scripts/amplification_analysis.py` — layer-wise adversarial amplification (`||Δa_l||_2`, ratio `R_l`).
- `scripts/verify_svd_projection.py` — verify a checkpoint satisfies `sigma_max(W) <= cap + 1e-5`.
