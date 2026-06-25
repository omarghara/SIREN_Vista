# Deep Research Prompt: CIFAR-10 SIREN/Functa Robustness

Copy/paste the prompt below into ChatGPT Deep Research.

---

I am working on a master's thesis project about adversarial robustness of
parameter-space classifiers built from implicit neural representations (INRs),
especially SIREN/Functa-style models. I need a deep literature search and a
research plan. Please act as an expert researcher in adversarial robustness,
implicit neural representations, meta-learning, neural fields, Lipschitz neural
networks, and bilevel optimization.

GitHub repository:

```text
https://github.com/omarghara/SIREN_Vista
```

Use the repository as extra implementation context if it is accessible, but do
not rely only on the code. The main task is to search the literature and propose
research directions. If you inspect the repo, focus on the CIFAR-10 Spatial
Functa pipeline, the SIREN variants, the PGD attack code, the inner phi-fitting
logic, and the warm-start regularizer experiments.

## Project Goal

The goal is to understand whether a Lipschitz-constrained SIREN/INR backbone can
make a CIFAR-10 parameter-space classifier more robust to PGD attacks.

The pipeline is:

```text
CIFAR-10 image x
  -> fit a compact spatial modulation phi for x against a shared SIREN/INR
  -> classify phi with a downstream classifier
  -> attack the full image-to-phi-to-classifier pipeline with PGD
```

I want to find papers, methods, and experiment ideas that can help me understand:

- how SIRENs and INRs work and why they can represent images
- why parameter-space or weight-space classifiers may appear robust
- how to evaluate these systems without falling into gradient masking
- how to make the fitted phi representation or the classifier on phi genuinely
  robust
- whether Lipschitz constraints, spectral constraints, orthogonality,
  Jacobian penalties, robust optimization, or adversarial training are promising
  here

## Current Project Context

The repository is called SIREN_Vista. The current focus is CIFAR-10.

At first, fitting a standard/global SIREN/Functa representation on CIFAR-10 was
not easy. We tried several variants such as SIREN, FINER, Fourier-style
features, and other INR variants. The breakthrough came after using ideas from
the paper "From Data to Functa" (arXiv:2302.03130), specifically spatially
arranged latent modulations. After implementing a Spatial Functa-style SIREN,
we finally got good image reconstruction quality and a usable classifier.

Current CIFAR Spatial Functa setup:

- dataset: CIFAR-10, RGB images, 32 x 32
- shared INR backbone: SIREN
- hidden width: 256
- SIREN depth: 6 sine layers
- SIREN frequency: 10.0
- spatial modulation phi: 8 x 8 x 16, total dimension 1024
- spatial lookup: nearest cell
- coordinates: local coordinates inside each cell
- modulation: shift-only modulation of SIREN hidden layers
- latent-to-shift map: linear map from phi cell vector to per-layer shifts
- downstream classifier: CNN over the fitted spatial phi grid
- current attack protocol: PGD-200 in image space, eps in {1,2,4,6,8}/255,
  while refitting phi with 5 inner optimization steps during the attack

The base vanilla model is a warm-start/meta-trained Spatial SIREN. It reached
good reconstruction quality and a CNN classifier around 76% clean top-1.

## Important Attack-Evaluation Lesson

We found an important mismatch/bug in the attack setup.

Originally, the attack was using more inner phi fitting steps and clipped
gradients during clean/final refitting. The functaset creation and classifier
training used only 3 or 5 inner steps and no phi-gradient clipping. This caused
a mismatch and artificially lowered the clean accuracy of the softlip model
during PGD, making some numbers invalid.

We patched the PGD attack so that:

- the attack-time phi fitting uses the same number of inner steps as the
  functaset/classifier setup
- current matched runs use 5 inner phi steps
- clean/final phi refit does not clip phi gradients, matching makeset.py

So the current trustworthy comparison is the matched inner-5, no-refit-clipping
PGD-200 evaluation.

## Current Main Results: Vanilla vs Softlip Tiered

The most relevant baseline comparison is:

- vanilla e512 Spatial SIREN
- softlip tiered e12 Spatial SIREN
- both use matched inner-5 functasets
- both use newly trained CNN classifiers over phi
- both attacked with patched PGD-200 using 5 inner phi steps
- n = 1000 CIFAR-10 test images

Classifier validation:

- vanilla inner5 CNN: 76.27% best logged top-1
- softlip tiered inner5 CNN: 75.73% best logged top-1

PGD-200 results on first 1000 CIFAR-10 test images:

| eps (/255) | vanilla clean | vanilla robust | softlip clean | softlip robust |
|---:|---:|---:|---:|---:|
| 1 | 0.761 | 0.527 | 0.763 | 0.534 |
| 2 | 0.761 | 0.312 | 0.763 | 0.319 |
| 4 | 0.761 | 0.044 | 0.763 | 0.064 |
| 6 | 0.761 | 0.003 | 0.763 | 0.005 |
| 8 | 0.761 | 0.003 | 0.763 | 0.001 |

Interpretation:

- Softlip tiered is nearly clean-matched to vanilla.
- It gives a small robustness edge at eps 1/255, 2/255, and especially 4/255.
- The clearest edge is eps 4/255: 64/1000 robust for softlip versus 44/1000
  for vanilla.
- By eps 6/255 and 8/255, both models are essentially broken.
- This is a positive signal, but it is not thesis-ready or a strong robustness
  result yet.

## What "Softlip Tiered" Means Here

The CIFAR softlip tiered model is not a simple global L/frequency cap. It is a
vanilla-relative spectral penalty:

- first sine layer: about 90% of vanilla spectral norm
- next two sine layers: about 85% of vanilla spectral norm
- later sine layers: about 75% of vanilla spectral norm
- readout: about 80% of vanilla spectral norm
- modulation map phi -> shifts: not capped in this experiment

Measured singular-value ratios confirmed this pattern approximately. This
reduced the worst-case SIREN amplification bound from roughly 28782 for vanilla
to roughly 6846 for softlip tiered.

## New Warm-Start Experiments We Tried

We then tried several new regularizer experiments by warm-starting from the
vanilla e512 checkpoint and meta-training further. The idea was to avoid training
from scratch while testing ways to constrain the SIREN.

Experiments:

1. Orthogonality penalty:
   - Penalize selected linear layers with mean((G - I)^2).
   - For square layers, G = W^T W.
   - For wide layers such as RGB readout, use the feasible row form W W^T.
   - Applied to sine layers and readout.

2. Readout spectral caps:
   - cap only the final hidden-to-RGB readout layer
   - caps at 90%, 50%, and 10% of the vanilla readout spectral norm

3. Final sine layer cap:
   - cap the final sine layer before the RGB readout
   - cap at 10% of the vanilla final sine spectral norm

4. Counter-amplification caps:
   - compute the product amplification through the vanilla SIREN
   - try to counter the whole amplification by capping either:
     - only the RGB readout, or
     - only the final sine layer before readout

Warm-start PGD results were only on n = 200 images, so they are weaker evidence.
The completed 76%-class warm-start variants did not clearly beat softlip tiered.

Warm-start PGD-200, n = 200:

| model | clean | eps1 robust | eps2 robust | eps4 robust | eps6 robust | eps8 robust |
|---|---:|---:|---:|---:|---:|---:|
| vanilla inner5 | 0.790 | 0.540 | 0.280 | 0.040 | 0.000 | 0.000 |
| softlip tiered inner5 | 0.820 | 0.575 | 0.315 | 0.065 | 0.005 | 0.000 |
| warm_readout_cap10_lam1 | 0.790 | 0.530 | 0.320 | 0.040 | 0.000 | 0.000 |
| warm_prereadout_cap10_lam1 | 0.815 | 0.545 | 0.345 | 0.020 | 0.000 | 0.000 |
| warm_readout_cap50_lam1 | 0.830 | 0.500 | 0.310 | 0.040 | 0.005 | 0.005 |
| warm_prereadout_counter1_lam1e-2 | 0.840 | 0.535 | 0.280 | 0.040 | 0.000 | 0.000 |
| warm_orth_lam1e-3 | 0.770 | 0.490 | 0.260 | 0.010 | 0.005 | 0.000 |
| warm_readout_counter1_lam1e-2 | 0.720 | 0.515 | 0.395 | 0.070 | 0.025 | missing |

Important caveat:

- warm_readout_counter1_lam1e-2 looks most robust at eps 2, 4, and 6, but its
  classifier was interrupted after epoch 10 and only reached about 70.20%
  validation top-1, so it is not a fair apples-to-apples result.

## Verification of Whether the New Constraints Actually Converged

We created a diagnostic notebook to calculate exact SVD-based penalties and
signal amplification:

- exact spectral norms of all sine layers and readout
- cap violation: max(0, sigma(W) - cap)
- orthogonality penalty: mean((G - I)^2)
- cumulative Lipschitz-style amplification bound through the SIREN
- actual activation RMS through the SIREN for zero phi

Key verification conclusions:

- softlip tiered mostly achieved its intended layerwise caps
  - max cap violation around 0.00593
  - final product bound around 6846
- readout 90% cap basically achieved the cap
  - readout sigma 0.164538 vs cap 0.164199
  - but the checkpoint only reached epoch 1, so not useful as a completed model
- readout 50% cap did not fully reach its cap
  - sigma 0.101600 vs cap 0.091222
- readout 10% cap did not reach its cap
  - sigma 0.082407 vs cap 0.018244
- final-sine 10% cap did not fully reach its cap
  - sigma 0.085869 vs cap 0.061275
- counter-amplification caps were extremely tiny and did not converge
  - readout counter cap about 6.34e-06, actual sigma 0.136925
  - final-sine counter cap about 2.13e-05, actual sigma 0.184689
- orthogonal run did not really make the whole SIREN orthogonal
  - hidden square layers had small penalties
  - first sine layer dominated the penalty around 82.27
  - readout also had a nontrivial penalty around 0.322

Amplification final product bounds:

| model | cumulative product bound |
|---|---:|
| vanilla_base | 28782 |
| softlip_tiered_e12 | 6846 |
| warm_prereadout_cap10_lam1 | 6747 |
| warm_prereadout_counter1_lam1e-2 | 10895 |
| warm_readout_cap10_lam1 | 13284 |
| warm_readout_cap50_lam1 | 15790 |
| warm_readout_counter1_lam1e-2 | 20767 |
| warm_orth_lam1e-3 | 22138 |
| warm_readout_cap90_lam1 | 25249 |

Interpretation:

- Reducing the amplification bound is measurable.
- But the warm-start variants did not clearly improve PGD robustness.
- This suggests either:
  - the product spectral bound is too loose,
  - the wrong part of the system is being constrained,
  - the phi-fitting map x -> phi is the real sensitivity bottleneck,
  - the downstream classifier over phi is too fragile,
  - or the attack/optimization dynamics dominate the behavior.

## What I Need From Deep Research

Please search the literature and give me a structured, cited research report.
Prioritize papers with direct relevance, but include adjacent papers if they
suggest useful methods.

I want the report to answer these questions:

### 1. Background and mechanisms

- What are the most important papers for understanding SIREN, INRs, Fourier
  features, FINER-like variants, spatial Functa, and learned modulations?
- Why do spatial latent grids help scale Functa/INRs to CIFAR-like images?
- What does the literature say about the stability or conditioning of SIRENs
  with periodic activations and frequency scaling?

### 2. Weight-space / parameter-space classifiers and robustness

- What papers study classifiers that operate on fitted weights, INR
  parameters, modulations, hypernetwork latents, or learned function codes?
- What papers attack these systems?
- What are the best known adaptive attack methods for systems with an inner
  optimization loop?
- How should I evaluate robustness without mistaking gradient masking for real
  robustness?
- Please include papers on BPDA, EOT, unrolled optimization attacks, implicit
  differentiation, bilevel attacks, and gradient-obfuscation diagnostics if
  relevant.

### 3. Lipschitz and spectral constraints

- What papers are most relevant for constraining neural networks with spectral
  normalization, Parseval/orthogonal constraints, Bjork orthonormalization,
  Cayley/orthogonal layers, Jacobian regularization, gradient penalties, or
  certified Lipschitz networks?
- Which of these methods are practical for SIRENs or INRs?
- What are the pitfalls of simply minimizing product-of-spectral-norm bounds?
- Are there better layerwise or local Jacobian constraints for periodic
  activations?

### 4. What part of our pipeline should be constrained?

Given our results, please analyze which component is likely the true
robustness bottleneck:

- the shared SIREN backbone f_theta
- the modulation map phi -> shifts
- the inner fitting map x -> phi*(x)
- the downstream classifier g(phi)
- the full composed map x -> phi*(x) -> g(phi*(x))

I especially want ideas for regularizers or objectives that target:

- d output / d phi
- d logits / d phi
- d phi*(x) / d x
- d logits / d x through the fitted phi
- stability of the inner optimizer
- classifier margin in phi-space
- smoothness of phi across the 8 x 8 spatial grid

### 5. Concrete next experiments

Please propose a ranked list of concrete experiments I can run next. For each
experiment, include:

- the idea
- which part of the pipeline it targets
- why it might work better than the experiments I already tried
- likely implementation cost
- expected failure modes
- how to evaluate it fairly

Please include ideas in these categories:

- better SIREN/INR regularizers
- better phi-space classifier training
- adversarial training in image space or phi space
- meta-training objectives that encourage stable phi fitting
- inner-loop regularization or smoothing
- spatial regularizers on phi
- attack-strength checks and gradient-masking diagnostics
- architecture changes to the modulation map or classifier

### 6. Evaluation protocol

Please propose a robust, defensible evaluation protocol for this thesis:

- what attack variants to run
- how many images to use at each stage
- how to match clean accuracy fairly
- how to test for gradient masking
- how to compare vanilla vs constrained models
- what ablations are essential
- what diagnostics would support or refute the claim that the model is truly
  more robust

### 7. Deliverables I want in your answer

Please structure the final answer as:

1. Executive summary
2. Annotated bibliography with citations and links
3. Concepts I need to understand, explained clearly
4. Diagnosis of my current results
5. Ranked next experiments
6. Recommended attack/evaluation protocol
7. Open risks and what would invalidate the thesis claim

Please be concrete and critical. Do not just say "use adversarial training" or
"use spectral normalization" generically. Tie each recommendation to this exact
pipeline:

```text
x -> fit spatial phi with inner optimization -> CNN classifier over phi
```

If there are no direct papers for this exact setup, say that clearly and use the
closest adjacent literature, explaining the inference you are making.

Please cite primary sources whenever possible and include links.
