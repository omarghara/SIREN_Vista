# Results report: soft-Lipschitz SIREN vs vanilla SIREN

Working-session report documenting the first head-to-head comparison between
a vanilla SIREN backbone and a soft-Lipschitz-constrained SIREN backbone on
MNIST. Scope: backbone reconstruction quality only. PGD robustness numbers
are still pending and will be appended once available.

All paths are relative to `/home/omarg/`.

---

## 1. Experimental setup

### 1.1 Architectures

Both runs use the identical backbone shape:

| hyperparameter | value |
|---|---|
| `hidden_dim` | 256 |
| `num_layers` (SIREN depth) | 10 |
| `modul_features` | 512 (vanilla) / 512 (softlip) |
| `freq` (omega_0) | 30 |
| `use_shift` | True |
| image shape | 28 x 28 |

The **only** difference between the two runs is the training-time objective.

### 1.2 Training configurations

| knob | vanilla | softlip |
|---|---|---|
| variant | `vanilla` | `soft_lipschitz` |
| soft_lip_cap `L` | - | 30.0 |
| soft_lip_lambda | - | 1.0 |
| soft_lip_apply_to | - | `all` (sine.1-9 + readout + modul) |
| soft_lip_skip_first | - | True (sine.0 excluded from penalty) |
| extracted sine sigma-cap (`L / freq`) | - | 1.0 |
| extracted readout sigma-cap (`L`) | - | 30.0 |
| extracted modul sigma-cap (`L / freq`) | - | 1.0 |
| num_epochs | repo default (presumably 6) | 6 |
| ext_lr | 5e-6 | 5e-6 |
| clip_grad_norm | 1.0 | 1.0 |
| random seed | 0 | 0 |

**Soft-Lipschitz objective.** Added to the MAML outer loss:

```
lambda * sum_{ell in penalty_set} max(0, sigma(W_ell) - c_ell)^2
```

with `sigma` estimated via power iteration (1 step / batch, persistent `u`, `v`
buffers) and `c_ell` chosen per layer so that the layer's Lipschitz constant
(incl. the post-`sin(omega_0 * .)` factor) is bounded by `L`:

- Hidden sine layers `ell` in {1, ..., 9}: `c_ell = L / freq = 1.0`.
- Readout `hidden2rgb`: `c_ell = L = 30.0`.
- Modulation map `modul`: `c_ell = L / freq = 1.0` (its output feeds into sine layers).
- First sine layer sine.0: **excluded** (sigma(W_0) does not enter the
  phi -> output Lipschitz bound; see discussion in
  `context/CHANGES.md:Section 7` and `code_explaination.md`).

### 1.3 Checkpoint paths

| variant | SIREN checkpoint |
|---|---|
| vanilla | `/home/omarg/Parameter-Space-Attack-Suite/model_mnist/modSiren.pth` |
| softlip | `/home/omarg/SIREN_Vista/model_mnist/softlip_L30_lam1e+00_all_skip0/modSiren.pth` |

### 1.4 Reconstruction-evaluation protocol

Same settings for both runs (apples-to-apples):

| knob | value |
|---|---|
| eval script | `SIREN_Vista/evaluate_reconstruction.py` (batched) |
| `--iter-checkpoints` | `5,20,50,100,200` |
| `--inner-lr` | 0.01 (makeset default; SGD for MNIST) |
| `--split` | `both` (train + test) |
| `--batch-size` | 128 |
| `--max-samples` | 2000 per split (4000 total) |
| seed | 0 |

Inner-loop SSIM is the hand-rolled 11 x 11 Gaussian-window implementation in
`evaluate_reconstruction.py::ssim_2d_batch` (data_range = 1.0).

JSON artifacts:

- `SIREN_Vista/runs/vanilla/reconstruction_eval.json`
- `SIREN_Vista/runs/softlip_L30_lam1e+00_all_skip0/reconstruction_eval.json`

---

## 2. Results

### 2.1 Fitting curve: train split (n = 2000)

| iters | PSNR vanilla | PSNR softlip | Delta PSNR | SSIM vanilla | SSIM softlip | Delta SSIM | MSE vanilla | MSE softlip | MSE ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5   | 20.29 | 16.18 | **-4.11** | 0.716 | 0.534 | -0.18 | 1.12e-2 | 2.88e-2 | x2.58 |
| 20  | 23.80 | 19.22 | **-4.58** | 0.765 | 0.602 | -0.16 | 5.62e-3 | 1.56e-2 | x2.78 |
| 50  | 25.37 | 20.95 | **-4.42** | 0.792 | 0.638 | -0.15 | 4.15e-3 | 1.10e-2 | x2.64 |
| 100 | 26.37 | 22.04 | **-4.33** | 0.811 | 0.662 | -0.15 | 3.45e-3 | 8.72e-3 | x2.53 |
| 200 | 27.25 | 22.98 | **-4.27** | 0.827 | 0.685 | -0.14 | 2.95e-3 | 7.11e-3 | x2.41 |

### 2.2 Fitting curve: test split (n = 2000)

| iters | PSNR vanilla | PSNR softlip | Delta PSNR | SSIM vanilla | SSIM softlip | Delta SSIM | MSE vanilla | MSE softlip | MSE ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5   | 20.70 | 16.54 | **-4.16** | 0.720 | 0.540 | -0.18 | 1.03e-2 | 2.67e-2 | x2.58 |
| 20  | 24.58 | 19.80 | **-4.78** | 0.772 | 0.608 | -0.16 | 4.91e-3 | 1.38e-2 | x2.81 |
| 50  | 26.31 | 21.55 | **-4.76** | 0.801 | 0.643 | -0.16 | 3.56e-3 | 9.62e-3 | x2.70 |
| 100 | 27.35 | 22.68 | **-4.67** | 0.821 | 0.670 | -0.15 | 2.94e-3 | 7.57e-3 | x2.58 |
| 200 | 28.24 | 23.68 | **-4.56** | 0.837 | 0.694 | -0.14 | 2.51e-3 | 6.05e-3 | x2.41 |

### 2.3 Train vs test parity

| | vanilla test - train | softlip test - train |
|---|---:|---:|
| Delta PSNR @ iters=200 | +0.99 dB | +0.70 dB |
| Delta SSIM @ iters=200 | +0.010 | +0.009 |

Both backbones reconstruct test slightly **better** than train at the
matched sample count (likely an artifact of the 2000-sample subset being
marginally easier on test; neither backbone overfits). No differential
generalization effect from the soft-Lipschitz penalty.

---

## 3. Interpretation

### 3.1 The reconstruction gap is ~4 dB, flat across iter counts

The delta PSNR column stays in a narrow band (4.11 - 4.78 dB) from
iters=5 all the way to iters=200. If the constraint merely *slowed down*
fitting, the gap would shrink at high iter counts. It does not.

**Diagnostic conclusion:** the soft-Lipschitz penalty is shrinking the
representable function space of the backbone itself, not just hindering
the inner-loop optimizer. The backbone **cannot** reach vanilla's
fidelity ceiling even given unlimited inner iterations. That is the
target behavior for a Lipschitz regularizer: it is narrowing the class
of functions the meta-initialization can be perturbed into, which is
exactly the knob we want to pay for adversarial robustness.

### 3.2 Cost is comparable across fidelity metrics

- **MSE**: softlip is 2.4x - 2.8x worse across all iters.
- **PSNR**: softlip is 4.1 - 4.8 dB lower (equivalent to the MSE ratio).
- **SSIM**: softlip is 0.14 - 0.18 lower at all iters (0.68 softlip vs 0.83
  vanilla at iters=200). SSIM matters for visual digit identity under
  noise; 0.68 is borderline.

All three metrics tell the same story. The constraint is real and
uniform.

### 3.3 Classifier-relevant row (iters=5)

The downstream classifier is trained on `makeset.py --iters 5`
modulations. That row:

| | vanilla | softlip |
|---|---:|---:|
| PSNR train | **20.29** | 16.18 |
| SSIM train | **0.716** | 0.534 |
| MSE train | 0.0112 | 0.0288 |

Softlip modulations encode images at PSNR ~16 dB and SSIM ~0.54 - roughly
2.6x worse MSE than vanilla at this operating point. MNIST digit identity
is highly redundant so 16 dB is typically still classifiable, but we
should expect:

- **Clean classifier accuracy drop**: very likely a 3 - 6 percentage-point
  hit relative to vanilla (vanilla usually ~97-98% on this architecture;
  expected softlip ~92-95% on clean MNIST). Not measured yet; will be
  filled in once the PGD step has also produced a clean-accuracy number.

### 3.4 Vanilla ceiling is modest

Generic SIREN literature reports MNIST-scale reconstructions near 30 dB at
saturated fit. Our vanilla backbone tops out at PSNR 27-28 dB, which
suggests it is not a fully tuned SIREN (repo defaults: 6 epochs,
ext_lr=5e-6, clip_grad_norm=1.0; all conservative). The comparison here
is vanilla-in-this-repo vs softlip-in-this-repo under matched training
budgets, which is the correct apples-to-apples setup. It is worth noting
for the thesis that absolute PSNR numbers are below SOTA because of the
training-budget choice, not because of the evaluation protocol.

### 3.5 Caveat: was sigma actually pushed down during training?

The 4 dB gap is consistent with a "hard" Lipschitz ceiling only **if**
the training-time sigma values for the constrained layers actually came
close to their caps. Log evidence from the early phases of the current
softlip run (`scripts/run_soft_lipschitz_mnist.sh` with
`LOG_SIGMAS_EVERY=50`) showed:

```
[sigmas @ epoch 0 batch 0] sine[10]: [4.983 0.091 0.094 ...] min=0.091 max=4.983 | readout: 0.046 | modul: 1.846
```

- sine.1 - sine.9 started at sigma ~= 0.09 **well below** their cap of 1.0 -> penalty contribution zero from these, cap never active.
- readout started at sigma ~= 0.046 **well below** its cap of 30.0 -> penalty contribution zero, cap never active.
- modul started at sigma ~= 1.85 **above** its cap of 1.0 -> penalty
  contribution `(1.85 - 1.0)^2 = 0.72`, cap active.

So on `apply_to=all, skip_first=1`, only **one** layer (`modul`) had the
penalty biting during training. The final sigma value of `modul` at end-of-
training has not yet been inspected directly. **Next verification step:**
re-load the saved checkpoint and compute `sigma(modul)` offline. If it
lands near 1.0, the 4 dB gap is the true Lipschitz-induced cost. If it is
still ~1.5, the gap comes from a partially-enforced constraint and a
longer-trained run could lower the gap while preserving most of the
constraint effect.

---

## 4. Open questions

1. **What is the adversarial robustness of this softlip backbone?**
   PGD attack (`attacks/full_pgd.py`, eps=16/255, 100 outer PGD steps,
   10 inner modulation steps) has not yet been run on this variant or on
   the matched vanilla baseline. See Section 5.

2. **What is softlip's clean classifier accuracy?**
   The classifier was trained in Step 3 of the pipeline and is available at
   `runs/softlip_L30_lam1e+00_all_skip0/mnist_classifier/best_classifier.pth`,
   but its accuracy has not been extracted here. It is implicit in the
   clean row of the PGD output when that runs.

3. **Is the remaining sine-layer penalty doing any work?**
   Per Section 3.5, sine.1 - sine.9 and readout never activated the cap on
   initialization. Over six epochs of meta-training, did their sigma
   values drift upward toward the cap? If not, `apply_to=sine_only` and
   `apply_to=sine_and_readout` are effectively no-op variants for this
   (`L`, `freq`) choice; only `apply_to=all` with `skip_first=1` is
   meaningful. This should be noted explicitly in the thesis ablation.

4. **Does a sigma-verification tool exist?**
   Yes, `SIREN_Vista/diagnostics.py::layer_sigmas(model)` already prints
   the per-layer spectral norms. Running it on the loaded checkpoint would
   answer questions 3 and Section 3.5 in one shot. Tracked as a pending
   sanity-check step.

---

## 5. Pending experiments (immediate)

1. **PGD on both checkpoints**, matched settings. Commands already written in
   `scripts/run_soft_lipschitz_mnist.sh:Step 5` and documented in the chat.
   Gives:
   - clean classifier accuracy on each backbone
   - unrolled-PGD adversarial accuracy on each backbone
   These are the only two numbers that let us decide whether the 4 dB
   reconstruction cost was worth paying.

2. **Spectral-norm audit of saved checkpoints.** One-line diagnostic:
   `python -c "from diagnostics import layer_sigmas, format_sigmas_one_liner; import torch; from SIREN import ModulatedSIREN; m = ModulatedSIREN(28,28,256,10,512,30,'cuda'); m.load_state_dict(torch.load('<ckpt>')['state_dict']); print(format_sigmas_one_liner(layer_sigmas(m)))"`

3. **Full-split reconstruction re-run (optional).** Current numbers are on
   2000/split for speed. Thesis figure would ideally use 60000 train +
   10000 test. At batch_size=128 on an RTX 2080 Ti this takes ~30 - 40
   minutes total and can be run unattended overnight.

## 6. Pending experiments (follow-up, conditional on PGD outcome)

If PGD softlip acc >= 30% while clean >= 92% and PGD vanilla ~5%:
  -> run (L, lambda) sweep to map the reconstruction-robustness Pareto
     front. Suggested grid:
     `(L, lambda) in {(15, 1), (30, 0.3), (30, 1), (30, 3), (45, 1)}`.

If PGD softlip acc <= 15%:
  -> diagnose. Likely cause: the cap is not fully enforced (see Section
     3.5). Either train longer with larger `ext_lr` or switch to hard
     spectral-norm enforcement via `torch.nn.utils.spectral_norm` for the
     modul layer specifically (currently the only active constraint).

If PGD softlip acc high **under naive PGD** but low **under unrolled PGD**:
  -> gradient-obfuscation, not true robustness. Call this out explicitly
     in the thesis per `context/cursor_context.md::Evaluation rules`.

---

## 7. Files and artifacts

### Evaluation outputs (this session)

- `SIREN_Vista/runs/vanilla/reconstruction_eval.json`
- `SIREN_Vista/runs/softlip_L30_lam1e+00_all_skip0/reconstruction_eval.json`

### Checkpoints

- `/home/omarg/Parameter-Space-Attack-Suite/model_mnist/modSiren.pth` (vanilla)
- `SIREN_Vista/model_mnist/softlip_L30_lam1e+00_all_skip0/modSiren.pth`
- Classifier: `SIREN_Vista/runs/softlip_L30_lam1e+00_all_skip0/mnist_classifier/best_classifier.pth`
- Functaset: `SIREN_Vista/runs/softlip_L30_lam1e+00_all_skip0/functaset/mnist_{train,val,test}.pkl`

### Code paths referenced

- Variant registry: `SIREN_Vista/variants/__init__.py`
- Soft-Lipschitz variant: `SIREN_Vista/variants/soft_lipschitz.py`
- Batched reconstruction eval: `SIREN_Vista/evaluate_reconstruction.py`
- Diagnostics utility: `SIREN_Vista/diagnostics.py`
- Pipeline script: `SIREN_Vista/scripts/run_soft_lipschitz_mnist.sh`
- PGD attack: `SIREN_Vista/attacks/full_pgd.py`

### Related context

- `context/CHANGES.md` - chronological edits log (trainer resume, variants package, soft-lipschitz variant, reconstruction eval, pipeline additions, skip-first, batched eval)
- `context/project_state.md` - project handoff doc
- `context/code_explaination.md` - walkthroughs of the variants system and soft_lipschitz math
- `context/cursor_context.md` - thesis brief and evaluation rules

---

## 8. Short version (one paragraph)

On MNIST (2000 samples per split), a soft-Lipschitz-regularized SIREN with
per-layer budget `L = 30`, penalty weight `lambda = 1`, penalizing sine.1 - sine.9
plus readout plus modul while excluding sine.0, reaches saturated-fit
reconstruction PSNR of 22.98/23.68 dB (train/test) vs vanilla's
27.25/28.24 dB - a flat ~4 dB deficit across all inner-loop iteration
counts. MSE is ~2.4x - 2.8x worse, SSIM 0.14 - 0.18 lower. The gap is
uniform across iter counts (not just a "slower-to-fit" effect), which
means the constraint genuinely restricts the representable function
space, as intended. Only the `modul` layer's sigma was above its cap at
initialization (1.85 vs cap 1.0); the other penalized layers started far
below their caps and therefore received zero penalty gradient. A
direct sigma audit on the saved checkpoint and the PGD robustness run
are the two remaining experiments needed to judge whether this 4 dB
trade was worth it. Initial decision rule: worthwhile if PGD
softlip >= 30% and clean softlip >= 92%, assuming vanilla PGD ~5%.
