# CIFAR-10 Spatial-Functa CNN PGD Summary

Model: `vanilla_spatial_siren_cnn`

Each row is one Full-PGD run of `attacks/full_pgd_cifar10_spatial.py`.

| eps (/255) | n | clean acc | robust acc | robust \| clean | gap (clean-robust) |
|---:|---:|---:|---:|---:|---:|
| 8 | 200 | 0.7550 | 0.0050 | 0.0066 | +0.7500 |
| 16 | 200 | 0.7550 | 0.0050 | 0.0066 | +0.7500 |
| 32 | 200 | 0.7550 | 0.0050 | 0.0066 | +0.7500 |
| 64 | 200 | 0.7550 | 0.0050 | 0.0066 | +0.7500 |
