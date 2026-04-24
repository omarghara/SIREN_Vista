# Project Context for Cursor: INR Thesis on Parameter-Space Classifiers and Adversarial Robustness

## Main repo
- GitHub repository: https://github.com/tamirshor7/Parameter-Space-Attack-Suite

This repository is the official implementation for the paper **"Adversarial Robustness in Parameter-Space Classifiers"**. The repo README describes a 3-stage pipeline:
1. train a meta-optimized SIREN,
2. create a **Functaset** of SIREN modulation vectors,
3. train a **clean weight-space classifier** and then run the attack suite.

The README also lists the main datasets as **MNIST**, **Fashion-MNIST**, and **ModelNet10**.

---

## Thesis goal

This master's thesis is about **parameter-space classifiers built on implicit neural representations (INRs)**.

### Core pipeline
Instead of classifying directly in signal space (for example raw pixels), the pipeline is:

1. take an input signal/image,
2. fit or adapt an INR representation for it,
3. extract a parameter-space representation (typically SIREN modulation vectors or weights),
4. classify using a downstream classifier in **weight/modulation space**.

Symbolically:

```text
x (signal) -> INR fitting / modulation inference -> z*(x) (weights/modulations) -> classifier g(z*(x)) -> prediction
```

### Main problem
Recent work shows that these parameter-space classifiers may appear robust to standard white-box attacks, but a major part of this robustness is likely **not real robustness**. It often comes from:

- inner-loop optimization,
- difficult differentiation through INR fitting,
- gradient obfuscation / gradient masking,
- representation bottlenecks.

### Thesis objective
Move from **accidental robustness** to **structural / principled robustness**.

---

## The two main thesis directions

### 1. Lipschitz-bounded SIRENs for adversarial robustness
Main idea:
- the sine activation is bounded and smooth,
- but the **linear layers between sine activations** can still amplify perturbations,
- so we want to constrain or regularize those linear layers.

Potential methods:
- spectral norm control,
- orthogonal / near-orthogonal parameterizations,
- Parseval-style constraints,
- singular value regularization,
- Jacobian regularization,
- condition number control,
- explicit representation-stability regularization.

### 2. Extend beyond vanilla SIREN using Fourier features or richer INR representations
Main idea:
- current demos are mostly on simple benchmarks,
- vanilla SIRENs may not scale well to richer natural-image regimes,
- Fourier features or other richer INR encodings may improve fidelity and scalability.

Important caution:
- if current robustness partly comes from the INR acting like a bottleneck or low-pass filter,
- then improving the INR’s expressive power may reduce that apparent robustness.
- so richer INR representations are both a scaling opportunity and a robustness stress test.

---

## Core scientific questions

1. Is the observed robustness in parameter-space classifiers genuine or mostly gradient obfuscation?
2. If we constrain SIREN linear layers, do we improve **true end-to-end stability**, or do we only make gradients harder to compute?
3. Is **condition number control** actually enough?  
   Probably not by itself:
   - condition number controls relative stretching,
   - but it does **not** by itself bound absolute amplification,
   - so spectral norm / operator norm control is likely more directly tied to Lipschitz behavior.
4. Can richer INR representations such as Fourier features make the framework viable beyond toy datasets?
5. If we scale the INR, does the current apparent robustness disappear?

---

## Key conceptual warning for Cursor

Do **not** assume robustness just because PGD fails.

This project lives in a **bilevel / nested** pipeline. The attack is on the original signal, but classification happens after INR fitting or modulation inference. Therefore:
- naive white-box attacks can underestimate vulnerability,
- gradient masking is a serious risk,
- adaptive attacks are mandatory.

Any robustness claim in this project must be evaluated against the **full composed system**, not just the downstream classifier.

---

## Main papers to know

### 1. SIREN / INR foundation
- **Implicit Neural Representations with Periodic Activation Functions**
- Link: https://arxiv.org/abs/2006.09661

Why it matters:
- defines SIREN,
- explains periodic activations,
- discusses initialization and derivative behavior,
- gives the base INR architecture for the project.

### 2. Functa / learning on INR representations
- **From Data to Functa: Your Data Point Is a Function and You Can Treat It Like One**
- Link: https://arxiv.org/abs/2201.12204

Why it matters:
- shows how to treat each datapoint as a function,
- introduces modulation-based representations,
- gives the weight/modulation-space learning pipeline that this project builds on.

### 3. Weight-space background paper from the advisor list
- Link currently listed: https://arxiv.org/abs/1706.05806

Important note:
- this arXiv ID is actually **SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability**, not HyperNetworks.
- It may still be useful as an analysis/representation-comparison tool.
- If the advisor intended a true weight-generation / parameter-space background paper, the likely intended paper may have been:
  - **HyperNetworks**: https://arxiv.org/abs/1609.09106

### 4. Main attack paper this thesis builds on
- **Adversarial Attacks in Weight-Space Classifiers**
- OpenReview: https://openreview.net/forum?id=eOLybAlili

Why it matters:
- directly studies adversarial attacks on parameter-space classifiers,
- argues that much of the apparent robustness comes from gradient obfuscation,
- is the main paper this thesis should build from.

---

## Mathematical ideas Cursor should keep in mind

### Operator norm / spectral norm
For a matrix `W`, the spectral norm `||W||_2` is the maximum amount `W` can amplify an `l2` perturbation.

This is directly relevant because for layered networks, a rough Lipschitz upper bound is the product of layer operator norms.

### Condition number
`kappa(W) = sigma_max(W) / sigma_min(W)`

Important:
- condition number measures conditioning / relative distortion,
- but **does not** by itself ensure small absolute sensitivity,
- a matrix can have condition number 1 and still have huge norm,
- so condition number control alone is probably not enough for robustness.

### Composition of sensitivity
The real object of interest is not only the classifier `g`, but the full pipeline:

```text
x -> z*(x) -> g(z*(x))
```

So true robustness depends on both:
- sensitivity of `z*(x)` to input perturbations,
- sensitivity of `g` to changes in representation space.

### Implicit differentiation viewpoint
If `z*(x)` is the solution to an inner optimization problem, then the stability of `z*(x)` depends on the conditioning of that optimization problem and its Hessian / local curvature.

This makes conditioning, Jacobian control, and optimization stability central to the thesis.

---

## Repo structure that matters most

From the GitHub repo landing page, the key files/folders are:

- `attacks/`
- `SIREN.py`
- `trainer.py`
- `makeset.py`
- `train_classifier.py`
- `dataloader.py`
- `dataloader_modelnet.py`
- `utils.py`
- `environment.yml`

### Likely purpose of each file
- `SIREN.py`  
  Most likely defines the INR / SIREN architecture.  
  This is the first place to inspect for introducing:
  - spectral normalization,
  - orthogonal constraints,
  - custom parameterizations,
  - Jacobian-aware wrappers,
  - frequency / initialization modifications.

- `trainer.py`  
  Likely trains the meta-optimized SIREN backbone used before functaset creation.  
  This is a key place to apply:
  - regularization losses,
  - constrained optimization logic,
  - representation-stability penalties,
  - logging for conditioning / singular values.

- `makeset.py`  
  Likely builds the functaset / modulation dataset.  
  Important for understanding how signals are converted into parameter-space examples.

- `train_classifier.py`  
  Likely trains the downstream weight-space classifier.  
  Important for:
  - classification baselines,
  - evaluating whether robustness comes from the outer classifier or the inner INR stage,
  - adding outer-space regularization if needed.

- `attacks/`  
  Critical for this thesis.  
  This is where adaptive attack logic likely lives.  
  Any proposed defense must be tested here, not just on clean training metrics.

---

## Suggested research workflow for Cursor

### Step 1: Understand the baseline exactly
Cursor should first help map the current baseline pipeline:
1. how the SIREN is trained,
2. how modulations / functaset examples are generated,
3. how the classifier is trained,
4. how attacks are run,
5. where gradients flow and where they break.

### Step 2: Identify the real sensitivity bottlenecks
Questions to answer:
- Where does gradient obfuscation enter?
- Is it caused by optimization truncation?
- Is it caused by non-smooth steps or just bad conditioning?
- Is the representation itself stable, or only hard to differentiate through?

### Step 3: Start with the cleanest intervention
Best first intervention:
- constrain SIREN linear layers using **spectral normalization** or a similarly simple operator-norm bound.

Why start there:
- easy to justify,
- tied directly to worst-case amplification,
- easier to compare with baseline than more exotic ideas.

### Step 4: Compare against stronger alternatives
Then compare to:
- Parseval / orthogonality constraints,
- condition number regularization,
- Jacobian penalties,
- explicit representation-stability regularization.

### Step 5: Re-run strong attacks
Every intervention must be evaluated with:
- naive PGD only as a baseline,
- stronger adaptive attacks,
- attacks that reason about the inner loop,
- potentially BPDA-like approximations,
- black-box / gradient-free sanity checks if needed.

---

## Strong evaluation rules

Cursor should treat these as hard rules:

1. **Never trust robustness based only on vanilla PGD.**
2. **Always compare attack strength vs. robust accuracy curves.**
3. **Track both clean performance and reconstruction/fitting quality.**
4. **Measure representation stability directly**, not only final classification accuracy.
5. **Watch for compute-induced false confidence**:
   - a defense that makes attacks too expensive may look robust even when it is not.

---

## Useful experimental metrics

Besides clean and robust accuracy, track:
- INR reconstruction loss,
- functaset quality,
- modulation-space shift under perturbation,
- singular values / spectral norms of SIREN layers,
- condition surrogates,
- attack success rate under multiple attack strengths,
- convergence behavior of the inner fitting process,
- attack compute budget.

---

## Concrete hypotheses worth testing

### Hypothesis A
Bounding SIREN layer spectral norms reduces end-to-end sensitivity and improves real robustness under adaptive attacks.

### Hypothesis B
Condition number regularization alone is not sufficient, but combined with spectral norm control it may improve optimization stability and make implicit differentiation more reliable.

### Hypothesis C
Richer INR encodings (for example Fourier features) improve scalability but may reduce the accidental robustness currently caused by representation bottlenecks.

### Hypothesis D
The most principled target may be not only layer norms, but direct regularization of **representation stability**:
- force `z*(x)` and `z*(x + delta)` to stay close for small allowed perturbations.

---

## Good initial code-reading order

Recommended order for Cursor:
1. `README.md` in the repo
2. `SIREN.py`
3. `trainer.py`
4. `makeset.py`
5. `train_classifier.py`
6. `attacks/`
7. dataloaders + utils

This order mirrors the actual pipeline:
architecture -> INR training -> functaset creation -> classifier training -> attacks

---

## Good initial implementation order

Recommended order for modifying the project:
1. reproduce baseline clean pipeline,
2. reproduce at least one baseline attack result,
3. add spectral norm control to SIREN linear layers,
4. re-run clean training,
5. re-create functaset,
6. re-train classifier,
7. re-run attacks,
8. compare:
   - clean accuracy,
   - fitting quality,
   - robust accuracy,
   - representation stability,
   - attack behavior.

Only after that move to:
- Parseval / orthogonal methods,
- Jacobian regularization,
- condition number ideas,
- Fourier-feature variants.

---

## What Cursor should be careful not to do

- Do not confuse **conditioning** with **small Lipschitz constant**.
- Do not claim robustness from weak attacks.
- Do not assume SIREN bounded activations imply bounded network sensitivity.
- Do not optimize only for robustness if reconstruction / functaset quality collapses.
- Do not switch to richer INR encodings without checking whether apparent robustness changes.

---

## Short summary Cursor can keep in mind

This project is **not** just about making SIRENs more stable in isolation.

It is about making the full pipeline

```text
signal -> INR representation -> parameter-space classifier
```

more robust **for real**, under adaptive attacks that account for the inner optimization.

The key scientific challenge is separating:
- genuine structural robustness,
from
- robustness that only appears because gradients are hard to compute.

The cleanest first direction is:
- inspect the repo,
- reproduce the baseline,
- add spectral norm / Lipschitz-oriented control inside the SIREN,
- and test whether robustness survives stronger attacks.

---

## Optional next papers to consider after the main four

- Obfuscated Gradients Give a False Sense of Security  
  https://proceedings.mlr.press/v80/athalye18a.html

- Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks (AutoAttack)  
  https://arxiv.org/abs/2003.01690

- Parseval Networks: Improving Robustness to Adversarial Examples  
  https://arxiv.org/abs/1704.08847

- Spectral Normalization for GANs  
  https://arxiv.org/abs/1802.05957

- Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains  
  https://arxiv.org/abs/2006.10739

- NeRF  
  https://arxiv.org/abs/2003.08934

- Instant-NGP  
  https://arxiv.org/abs/2201.05989

---

## If updating this README later
Good future additions:
- exact command recipes you personally use,
- local directory layout on your machine,
- dataset paths,
- notes from reading each paper,
- known bugs / gotchas,
- attack results table,
- TODO list for each thesis direction.
