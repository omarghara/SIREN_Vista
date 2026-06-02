# Backbone checkpoint summary

- **path**: `model_cifar10/cifar10_spatial_warmvanilla_readout_counter1.0_lam1e-2/modSiren.pth`
- **model_name**: `cifar10_spatial_warmvanilla_readout_counter1.0_lam1e-2`
- **variant**: `spectral_cap`
- **epoch (best)**: 10
- **loss (best mean outer loss)**: 0.0015908752102404833
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

- `spec_cap_absolute`: None
- `spec_cap_counter_target`: 1.0
- `spec_cap_lambda`: 0.01
- `spec_cap_mode`: 'counter_amplification'
- `spec_cap_power_iters`: 10
- `spec_cap_reference_checkpoint`: 'model_cifar10/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/modSiren.pth'
- `spec_cap_scale`: 0.9
- `spec_cap_target`: 'readout'
