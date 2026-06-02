# Backbone checkpoint summary

- **path**: `model_cifar10/cifar10_spatial_warmvanilla_orth_sine_and_readout_auto_lam1e-3/modSiren.pth`
- **model_name**: `cifar10_spatial_warmvanilla_orth_sine_and_readout_auto_lam1e-3`
- **variant**: `orthogonal`
- **epoch (best)**: 19
- **loss (best mean outer loss)**: 0.08660641308797866
- **num_epochs requested**: 20
- **training epoch range**: 0–19

## model_args

- `coord_normalization`: 'zero_one_pixel_centers'
- `dataset`: 'cifar10'
- `depth`: 6
- `finer_first_bias_scale`: 1.0
- `finer_scale_req_grad`: False
- `fourier_include_input`: False
- `fourier_num_freqs`: 64
- `fourier_sigma`: 10.0
- `freq`: 10.0
- `height`: 32
- `hidden_dim`: 256
- `inner_lr`: 0.01
- `inner_optim`: 'sgd'
- `inner_steps`: 3
- `inr_type`: 'siren'
- `is_spatial`: True
- `latent_dim`: 16
- `latent_spatial_dim`: 8
- `lsa_include_linear`: True
- `lsa_init_scale`: 0.001
- `lsa_num_harmonics`: 8
- `mod_dim`: 1024
- `modulation_type`: 'shift'
- `out_features`: 3
- `phi_numel`: 1024
- `phi_shape`: (8, 8, 16)
- `spatial_interp`: 'nearest'
- `spatial_modulation`: True
- `use_local_coords`: True
- `width`: 32

## variant_args

- `orth_apply_to`: 'sine_and_readout'
- `orth_form`: 'auto'
- `orth_lambda`: 0.001
- `orth_skip_first`: False

## orthogonality diagnostics

Measured from the saved checkpoint on 2026-06-01.

The training penalty for this run was:

```text
L_orth = 1e-3 * sum_l mean((G_l - I)^2)
```

With `orth_form='auto'`, the sine affine layers use `G = W^T W`.
The RGB readout has shape `3 x 256`, so the run used `G = W W^T` for
that layer because `W^T W = I_256` is not feasible.

| layer | weight shape | Gram form | mean((G-I)^2) | weighted penalty | RMS(G-I) | Frobenius norm | max abs diff | sigma min | sigma max |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| sine0 | `(256, 2)` | `W^T W` | 82.267181 | 0.082267 | 9.070126 | 18.1403 | 13.6364 | 3.5559 | 3.8613 |
| sine1 | `(256, 256)` | `W^T W` | 0.003732 | 0.000004 | 0.061088 | 15.6384 | 0.9818 | 0.0002 | 0.4276 |
| sine2 | `(256, 256)` | `W^T W` | 0.003712 | 0.000004 | 0.060927 | 15.5973 | 0.9825 | 0.0004 | 0.5467 |
| sine3 | `(256, 256)` | `W^T W` | 0.003717 | 0.000004 | 0.060966 | 15.6073 | 0.9810 | 0.0000 | 0.4626 |
| sine4 | `(256, 256)` | `W^T W` | 0.003711 | 0.000004 | 0.060916 | 15.5945 | 0.9807 | 0.0004 | 0.4842 |
| sine5 | `(256, 256)` | `W^T W` | 0.003719 | 0.000004 | 0.060981 | 15.6110 | 0.9816 | 0.0002 | 0.6105 |
| readout | `(3, 256)` | `W W^T` | 0.321989 | 0.000322 | 0.567441 | 1.7023 | 0.9845 | 0.0814 | 0.1793 |

Totals for the exact training penalty:

```text
unweighted sum = 82.607760
weighted sum   = 0.082608
```

For reference, the final training log reported:

```text
epoch: 19, total: 0.086606, MSE: 0.001870, pen: 0.084737
```

Interpretation:

- Almost all of the saved checkpoint's orthogonality penalty comes from the first sine layer.
- The hidden `256 x 256` layers have small mean squared Gram errors, but they are not close to identity in a strong spectral sense. Their largest singular values are still below 1, and many singular values are close to 0.
- The `mean((W^T W - I)^2)` metric is diluted for large `256 x 256` Gram matrices, so it can look numerically small even when the layer is far from orthonormal.

Literal `W^T W - I` values for every layer, including the readout, are:

| layer | weight shape | mean((W^T W-I)^2) | weighted penalty | RMS | Frobenius norm | max abs diff |
|---|---:|---:|---:|---:|---:|---:|
| sine0 | `(256, 2)` | 82.267181 | 0.082267 | 9.070126 | 18.1403 | 13.6364 |
| sine1 | `(256, 256)` | 0.003732 | 0.000004 | 0.061088 | 15.6384 | 0.9818 |
| sine2 | `(256, 256)` | 0.003712 | 0.000004 | 0.060927 | 15.5973 | 0.9825 |
| sine3 | `(256, 256)` | 0.003717 | 0.000004 | 0.060966 | 15.6073 | 0.9810 |
| sine4 | `(256, 256)` | 0.003711 | 0.000004 | 0.060916 | 15.5945 | 0.9807 |
| sine5 | `(256, 256)` | 0.003719 | 0.000004 | 0.060981 | 15.6110 | 0.9816 |
| readout | `(3, 256)` | 0.003905 | 0.000004 | 0.062488 | 15.9968 | 1.0000 |
