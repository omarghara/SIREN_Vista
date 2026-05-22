# Training Stop Summary

- **model_name**: `functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000`
- **checkpoint**: `model_cifar10/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/modSiren.pth`
- **variant**: `vanilla`
- **requested epochs**: 512
- **stopped during**: epoch 278, around batch 78 / 391 (~20%)
- **saved best checkpoint epoch**: 277
- **saved best mean outer loss / MSE**: 0.0012658266946101737
- **stop reason**: manually stopped to preserve the improved checkpoint and free the machine for driver reboot / PGD evaluation.

## Recent Epochs

| epoch | terminal total | terminal MSE | penalty | note |
| ---: | ---: | ---: | ---: | --- |
| 275 | 0.001560 | 0.001560 | 0.000000 | completed |
| 276 | 0.001456 | 0.001456 | 0.000000 | completed |
| 277 | 0.001266 | 0.001266 | 0.000000 | completed; saved best checkpoint |
| 278 | ~0.0013 | ~0.0013 | 0.0000 | interrupted at ~20% |

## Recent Sigma Logs

The logged spectral norms were stable near the stopping point:

```text
[sigmas @ epoch 275 batch 200] sine[6]: [4.976 0.423 0.547 ...] min=0.423 max=4.976
[sigmas @ epoch 275 batch 250] sine[6]: [4.976 0.423 0.547 ...] min=0.423 max=4.976
[sigmas @ epoch 275 batch 300] sine[6]: [4.976 0.423 0.547 ...] min=0.423 max=4.976
[sigmas @ epoch 275 batch 350] sine[6]: [4.976 0.423 0.546 ...] min=0.423 max=4.976

[sigmas @ epoch 276 batch 0]   sine[6]: [4.976 0.423 0.547 ...] min=0.423 max=4.976
[sigmas @ epoch 276 batch 50]  sine[6]: [4.976 0.423 0.538 ...] min=0.423 max=4.976
[sigmas @ epoch 276 batch 100] sine[6]: [4.976 0.423 0.546 ...] min=0.423 max=4.976
[sigmas @ epoch 276 batch 150] sine[6]: [4.976 0.423 0.547 ...] min=0.423 max=4.976
[sigmas @ epoch 276 batch 200] sine[6]: [4.976 0.423 0.547 ...] min=0.423 max=4.976
[sigmas @ epoch 276 batch 250] sine[6]: [4.975 0.423 0.548 ...] min=0.423 max=4.975
[sigmas @ epoch 276 batch 300] sine[6]: [4.976 0.423 0.548 ...] min=0.423 max=4.976
[sigmas @ epoch 276 batch 350] sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976

[sigmas @ epoch 277 batch 0]   sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976
[sigmas @ epoch 277 batch 50]  sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976
[sigmas @ epoch 277 batch 100] sine[6]: [4.976 0.423 0.523 ...] min=0.423 max=4.976
[sigmas @ epoch 277 batch 150] sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976
[sigmas @ epoch 277 batch 200] sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976
[sigmas @ epoch 277 batch 250] sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976
[sigmas @ epoch 277 batch 300] sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976
[sigmas @ epoch 277 batch 350] sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976

[sigmas @ epoch 278 batch 0]  sine[6]: [4.976 0.423 0.549 ...] min=0.423 max=4.976
[sigmas @ epoch 278 batch 50] sine[6]: [4.976 0.423 0.548 ...] min=0.423 max=4.976
```

## Model Args Snapshot

- `dataset`: `cifar10`
- `inr_type`: `siren`
- `hidden_dim`: 256
- `mod_dim`: 1024
- `depth`: 6
- `freq`: 10.0
- `inner_optim`: `sgd`
- `inner_lr`: 0.01
- `inner_steps`: 3
- `spatial_modulation`: true
- `latent_spatial_dim`: 8
- `latent_dim`: 16
- `phi_shape`: `(8, 8, 16)`
- `phi_numel`: 1024
- `spatial_interp`: `nearest`
- `use_local_coords`: true
- `modulation_type`: `shift`

## Follow-up

Use this checkpoint for downstream makeset/classifier refresh or PGD evaluation after the NVIDIA driver/library mismatch is fixed.
