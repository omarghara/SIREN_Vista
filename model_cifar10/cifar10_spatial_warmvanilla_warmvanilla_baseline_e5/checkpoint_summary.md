# Backbone checkpoint summary

- **path**: `model_cifar10/cifar10_spatial_warmvanilla_warmvanilla_baseline_e5/modSiren.pth`
- **model_name**: `cifar10_spatial_warmvanilla_warmvanilla_baseline_e5`
- **variant**: `vanilla`
- **epoch (best)**: 4
- **loss (best mean outer loss)**: 0.0015582672079160278
- **num_epochs requested**: 5
- **training epoch range**: 0–4

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
