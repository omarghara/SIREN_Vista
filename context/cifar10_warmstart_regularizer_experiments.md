# CIFAR-10 Warm-Started SIREN Regularizer Experiments

Updated: 2026-06-02

## Purpose

These experiments test new ways to constrain the CIFAR-10 Spatial Functa
SIREN while avoiding the cost of training every backbone from scratch.

Instead of random initialization, each experiment starts from the already
trained vanilla CIFAR-10 Spatial Functa checkpoint:

```text
model_cifar10/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/modSiren.pth
```

The trainer uses:

```text
--init-from-checkpoint <vanilla checkpoint>
```

This loads only the model weights. It does not load the old optimizer state,
epoch counter, or best loss. So each run is a fresh meta-learning experiment
initialized from the vanilla backbone.

## Shared CIFAR Setup

All experiments use the same Spatial Functa architecture:

- dataset: CIFAR-10
- image size: `32 x 32 x 3`
- spatial phi grid: `8 x 8 x 16`
- total phi size: `1024`
- SIREN hidden width: `256`
- SIREN sine layers: `6`
- sine frequency: `10.0`
- spatial interpolation: nearest cell lookup
- coordinates: local coordinates inside each spatial cell
- modulation: shift-only
- inner phi fitting during meta-training: `3` SGD steps
- inner phi LR: `0.01`
- outer LR: `3e-05`

Launcher:

```text
scripts/run_cifar10_spatial_warmstart_regularizer.sh
```

Default output:

```text
model_cifar10/<experiment_name>/modSiren.pth
model_cifar10/<experiment_name>/logs/train.log
```

## Base Meta-Learning Loss

The original meta-learning objective is reconstruction MSE after inner-loop
phi fitting:

```text
L_base = MSE(INR(phi*(x)), x)
```

where `phi*(x)` is obtained by fitting phi for the image using the inner loop.

Every experiment below optimizes:

```text
L_total = L_base + L_regularizer
```

The trainer logs both pieces:

```text
total = MSE + pen
```

## Experiment 1: Orthogonal SIREN Weights

Command:

```bash
CUDA_GPU=0 EXPERIMENT=orthogonal \
  bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
```

Default settings:

```text
ORTH_LAMBDA=1e-3
ORTH_APPLY_TO=sine_and_readout
ORTH_FORM=auto
```

Regularizer:

```text
L_orth = lambda * sum_l mean((G_l - I)^2)
```

For square hidden SIREN layers:

```text
G_l = W_l^T W_l
```

So this is exactly the requested `W^T W - I` penalty.

For non-square layers, `ORTH_FORM=auto` chooses the feasible orthogonality
direction:

- if `out_dim >= in_dim`: use `W^T W - I`
- if `out_dim < in_dim`: use `W W^T - I`

This matters for the RGB readout, whose weight is `3 x 256`; `W^T W = I_256`
is impossible, so the default uses `W W^T = I_3` for that layer.

Default capped layers:

- all six SIREN sine affine layers
- final `hidden2rgb` readout
- not the modulation map

Optional variants:

```bash
# Only sine layers
ORTH_APPLY_TO=sine_only EXPERIMENT=orthogonal ...

# Include modulation map too
ORTH_APPLY_TO=all EXPERIMENT=orthogonal ...

# Force literal W^T W - I everywhere, even where impossible
ORTH_FORM=columns EXPERIMENT=orthogonal ...
```

## Experiment 2: Cap Only the Last RGB Readout

This constrains only:

```text
siren.hidden2rgb.weight
```

Regularizer:

```text
L_cap = lambda * max(0, sigma(W_readout) - c)^2
```

where `sigma(W)` is the largest singular value, estimated with power
iteration during training.

The vanilla readout spectral norm is:

```text
sigma(W_readout_vanilla) = 0.182443
```

### 2A: Readout Cap to 90 Percent

Command:

```bash
CUDA_GPU=0 EXPERIMENT=readout_cap90 \
  bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
```

Cap:

```text
c = 0.90 * 0.182443 = 0.164199
```

Loss:

```text
L_total = MSE + lambda * max(0, sigma(W_readout) - 0.164199)^2
```

### 2B: Readout Cap by 50 Percent

Command:

```bash
CUDA_GPU=1 EXPERIMENT=readout_cap50 \
  bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
```

Cap:

```text
c = 0.50 * 0.182443 = 0.091222
```

Loss:

```text
L_total = MSE + lambda * max(0, sigma(W_readout) - 0.091222)^2
```

### 2C: Readout Cap to 10 Percent

Command:

```bash
CUDA_GPU=0 SPEC_CAP_LAMBDA=1.0 EXPERIMENT=readout_cap10 \
  bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
```

Cap:

```text
c = 0.10 * 0.182443 = 0.018244
```

Loss:

```text
L_total = MSE + lambda * max(0, sigma(W_readout) - 0.018244)^2
```

This run is much more aggressive than the 90% and 50% readout caps.
The completed checkpoint is:

```text
model_cifar10/cifar10_spatial_warmvanilla_readout_cap10_lam1.0/modSiren.pth
```

### 2D: Final Sine Layer Cap to 10 Percent

This constrains the last sine affine layer before the RGB readout:

```text
siren.net.5.affine.weight
```

Command:

```bash
CUDA_GPU=1 SPEC_CAP_LAMBDA=1.0 EXPERIMENT=pre_readout_cap10 \
  bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
```

Cap:

```text
c = 0.10 * 0.612750 = 0.061275
```

Loss:

```text
L_total = MSE + lambda * max(0, sigma(W_sine_5) - 0.061275)^2
```

The completed checkpoint is:

```text
model_cifar10/cifar10_spatial_warmvanilla_prereadout_cap10_lam1.0/modSiren.pth
```

Default:

```text
SPEC_CAP_LAMBDA=1e-2
SPEC_CAP_POWER_ITERS=10
```

## Experiment 3: Counter the Whole SIREN Amplification in One Late Layer

This experiment measures the vanilla SIREN's layerwise amplification using
the product-style Lipschitz upper bound:

```text
A = product_i(freq * sigma(W_sine_i)) * sigma(W_readout)
```

For the current CIFAR SIREN:

```text
freq = 10.0
num sine layers = 6
```

Then we pick one late layer and make its cap small enough that the new product
bound would be approximately at most:

```text
COUNTER_TARGET
```

The default is:

```text
COUNTER_TARGET=1.0
```

This is intentionally aggressive and may hurt reconstruction. The cap scales
linearly with `COUNTER_TARGET`, so `COUNTER_TARGET=10` makes the cap 10x
larger.

### 3A: Counter Amplification in the Last RGB Readout

Command:

```bash
CUDA_GPU=0 EXPERIMENT=readout_counter COUNTER_TARGET=1.0 \
  bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
```

Cap formula:

```text
c_readout = COUNTER_TARGET / product_i(freq * sigma(W_sine_i))
```

For `COUNTER_TARGET=1.0`, the measured cap from the vanilla checkpoint is:

```text
c_readout = 6.3388e-06
```

Loss:

```text
L_total = MSE + lambda * max(0, sigma(W_readout) - 6.3388e-06)^2
```

This is much smaller than the current vanilla readout sigma `0.182443`, so
this run is expected to strongly pressure the readout and may degrade
reconstruction.

### 3B: Counter Amplification in the Layer Before RGB Readout

This caps the final sine affine layer:

```text
siren.net.5.affine.weight
```

Command:

```bash
CUDA_GPU=1 EXPERIMENT=pre_readout_counter COUNTER_TARGET=1.0 \
  bash scripts/run_cifar10_spatial_warmstart_regularizer.sh
```

Cap formula:

```text
c_pre_readout =
  COUNTER_TARGET /
  (sigma(W_readout) * freq * product_{i < last}(freq * sigma(W_sine_i)))
```

For `COUNTER_TARGET=1.0`, the measured cap from the vanilla checkpoint is:

```text
c_pre_readout = 2.1289e-05
```

The vanilla final sine-layer sigma is:

```text
sigma(W_sine_5_vanilla) = 0.612750
```

So this is also extremely aggressive.

## Useful Cap Measurements From Vanilla

Measured from the current vanilla checkpoint:

| target | vanilla sigma | 90% cap | 50% cap | 10% cap | counter cap, target=1 |
|---|---:|---:|---:|---:|---:|
| final sine layer `siren.net.5.affine` | `0.612750` | `0.551475` | `0.306375` | `0.061275` | `2.1289e-05` |
| RGB readout `hidden2rgb` | `0.182443` | `0.164199` | `0.091222` | `0.018244` | `6.3388e-06` |

## Completed Inner-5 Classifier and PGD Results

Updated 2026-06-02. These runs use the generic checkpoint pipeline:

```text
scripts/run_cifar10_spatial_inner5_checkpoint.sh
```

Protocol:

- make a new functaset with `5` inner phi SGD steps
- train the CNN classifier with the same best sweep settings used for the
  vanilla/softlip inner-5 comparison
- attack the first `200` CIFAR-10 test images with PGD-200
- use attack-time `--mod-steps 5`, matching the functaset/classifier
- evaluate eps `{1,2,4,6,8}/255` when complete

Classifier validation results:

| model | best val top-1 | note |
|---|---:|---|
| `warm_readout_cap10_lam1` | 76.86% | completed |
| `warm_prereadout_cap10_lam1` | 76.75% | completed |
| `warm_prereadout_counter1_lam1e-2` | 76.76% | completed |
| `warm_readout_cap50_lam1` | 76.42% | completed |
| `warm_orth_lam1e-3` | 76.27% | completed |
| `warm_readout_counter1_lam1e-2` | 70.20% | classifier was interrupted after epoch 10; use as provisional |
| `warm_readout_cap90_lam1` | missing | makeset was interrupted before classifier training |

PGD-200 results, `n=200`:

| model | eps (/255) | clean acc | robust acc | robust / clean |
|---|---:|---:|---:|---:|
| `warm_readout_cap10_lam1` | 1 | 0.790 | 0.530 | 0.671 |
| `warm_readout_cap10_lam1` | 2 | 0.790 | 0.320 | 0.405 |
| `warm_readout_cap10_lam1` | 4 | 0.790 | 0.040 | 0.0506 |
| `warm_readout_cap10_lam1` | 6 | 0.790 | 0.000 | 0.000 |
| `warm_readout_cap10_lam1` | 8 | 0.790 | 0.000 | 0.000 |
| `warm_prereadout_cap10_lam1` | 1 | 0.815 | 0.545 | 0.669 |
| `warm_prereadout_cap10_lam1` | 2 | 0.815 | 0.345 | 0.423 |
| `warm_prereadout_cap10_lam1` | 4 | 0.815 | 0.020 | 0.0245 |
| `warm_prereadout_cap10_lam1` | 6 | 0.815 | 0.000 | 0.000 |
| `warm_prereadout_cap10_lam1` | 8 | 0.815 | 0.000 | 0.000 |
| `warm_readout_cap50_lam1` | 1 | 0.830 | 0.500 | 0.602 |
| `warm_readout_cap50_lam1` | 2 | 0.830 | 0.310 | 0.373 |
| `warm_readout_cap50_lam1` | 4 | 0.830 | 0.040 | 0.0482 |
| `warm_readout_cap50_lam1` | 6 | 0.830 | 0.005 | 0.0060 |
| `warm_readout_cap50_lam1` | 8 | 0.830 | 0.005 | 0.0060 |
| `warm_prereadout_counter1_lam1e-2` | 1 | 0.840 | 0.535 | 0.637 |
| `warm_prereadout_counter1_lam1e-2` | 2 | 0.840 | 0.280 | 0.333 |
| `warm_prereadout_counter1_lam1e-2` | 4 | 0.840 | 0.040 | 0.0476 |
| `warm_prereadout_counter1_lam1e-2` | 6 | 0.840 | 0.000 | 0.000 |
| `warm_prereadout_counter1_lam1e-2` | 8 | 0.840 | 0.000 | 0.000 |
| `warm_orth_lam1e-3` | 1 | 0.770 | 0.490 | 0.630 |
| `warm_orth_lam1e-3` | 2 | 0.770 | 0.260 | 0.338 |
| `warm_orth_lam1e-3` | 4 | 0.770 | 0.010 | 0.0130 |
| `warm_orth_lam1e-3` | 6 | 0.770 | 0.005 | 0.0065 |
| `warm_orth_lam1e-3` | 8 | 0.770 | 0.000 | 0.000 |
| `warm_readout_counter1_lam1e-2` | 1 | 0.720 | 0.515 | 0.715 |
| `warm_readout_counter1_lam1e-2` | 2 | 0.720 | 0.395 | 0.549 |
| `warm_readout_counter1_lam1e-2` | 4 | 0.720 | 0.070 | 0.0972 |
| `warm_readout_counter1_lam1e-2` | 6 | 0.720 | 0.025 | 0.0347 |

Interpretation:

- The strongest clean classifier among these is `warm_readout_cap10_lam1`
  at 76.86% validation top-1, but its PGD robustness is close to vanilla.
- `warm_prereadout_cap10_lam1` is the best clean/robust compromise among the
  completed 76%-class classifiers at eps `2/255`, but it collapses by eps
  `4/255` to `0.020` robust accuracy.
- `warm_readout_counter1_lam1e-2` has the best robust / clean ratio and the
  best robust accuracy at eps `2/255`, `4/255`, and `6/255`, but this is not
  an apples-to-apples win because its classifier only reached 70.20% before
  interruption and eps `8/255` has not been run.
- None of these warm-start variants clearly dominates the earlier softlip
  tiered baseline yet.

## Recommended Run Order

Start with the less destructive experiments:

1. `readout_cap90`
2. `readout_cap50`
3. `orthogonal`
4. `readout_counter` with a larger `COUNTER_TARGET`, for example `10` or `100`
5. `pre_readout_counter` with a larger `COUNTER_TARGET`

The counter-amplification caps with `COUNTER_TARGET=1` are useful stress
tests, but they are probably too strong for preserving CIFAR reconstruction.

## Example Tmux Launches

Run two jobs on separate GPUs:

```bash
tmux new-session -d -s reg_readout_cap90_gpu0 \
  "cd /home/omarg/SIREN_Vista && CUDA_GPU=0 EXPERIMENT=readout_cap90 NUM_EPOCHS=20 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh"

tmux new-session -d -s reg_readout_cap50_gpu1 \
  "cd /home/omarg/SIREN_Vista && CUDA_GPU=1 EXPERIMENT=readout_cap50 NUM_EPOCHS=20 bash scripts/run_cifar10_spatial_warmstart_regularizer.sh"
```

Check progress:

```bash
tmux ls
tail -f model_cifar10/cifar10_spatial_warmvanilla_readout_cap90_lam1e-2/logs/train.log
```

## Implementation Files

- `trainer.py`
  - adds `--init-from-checkpoint`
  - initializes model weights from vanilla without optimizer/history
- `variants/orthogonal.py`
  - implements the `W^T W - I` / orthogonality penalty
- `variants/spectral_cap.py`
  - implements late-layer spectral cap and counter-amplification cap
- `scripts/run_cifar10_spatial_warmstart_regularizer.sh`
  - reproducible launcher for all experiments above

## Validation Status

The implementation was smoke-tested on CPU without starting training:

- Python syntax check passed for:
  - `trainer.py`
  - `variants/orthogonal.py`
  - `variants/spectral_cap.py`
  - `variants/__init__.py`
- `trainer.py --help` exposes the new flags.
- Variant registry includes:
  - `vanilla`
  - `soft_lipschitz`
  - `orthogonal`
  - `spectral_cap`
- The vanilla CIFAR Spatial Functa checkpoint loads strictly with
  `--init-from-checkpoint`.
- The orthogonal penalty returns a finite non-negative scalar.
- The spectral cap code returns the expected vanilla-derived caps:
  - readout 90%: `0.164199`
  - readout 50%: `0.091222`
  - readout counter target 1: `6.3388e-06`
  - pre-readout counter target 1: `2.1289e-05`
- `SPEC_CAP_POWER_ITERS` defaults to `10` so the 10% readout cap is active
  immediately when warm-starting from the vanilla checkpoint.
