# Backbone checkpoint summary

- **path**: `model_cifar10/cifar10_spatial_warmvanilla_svdproj_all_sine_readout_scale0.5_e5/modSiren.pth`
- **model_name**: `cifar10_spatial_warmvanilla_svdproj_all_sine_readout_scale0.5_e5`
- **variant**: `vanilla`
- **epoch (best)**: 4
- **loss (best mean outer loss)**: 0.005407449204892833
- **num_epochs requested**: 5
- **training epoch range**: 0–4

- **svd projection**: `all_sine_readout` (mode `reference_scale`, every 1)

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

## projection_args

- `cap`: None
- `cap_mode`: 'reference_scale'
- `enabled`: True
- `every`: 1
- `modul_cap`: None
- `readout_cap`: None
- `reference_checkpoint`: 'model_cifar10/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/modSiren.pth'
- `resolved_caps`: {'siren.net.0.affine.weight': 2.488150119781494, 'siren.net.1.affine.weight': 0.21168144047260284, 'siren.net.2.affine.weight': 0.2744978368282318, 'siren.net.3.affine.weight': 0.2299826443195343, 'siren.net.4.affine.weight': 0.2419731467962265, 'siren.net.5.affine.weight': 0.30637508630752563, 'siren.hidden2rgb.weight': 0.09122159332036972}
- `scale`: 0.5
- `sine_cap`: None
- `sine_freq_adjust`: True
- `target`: 'all_sine_readout'
