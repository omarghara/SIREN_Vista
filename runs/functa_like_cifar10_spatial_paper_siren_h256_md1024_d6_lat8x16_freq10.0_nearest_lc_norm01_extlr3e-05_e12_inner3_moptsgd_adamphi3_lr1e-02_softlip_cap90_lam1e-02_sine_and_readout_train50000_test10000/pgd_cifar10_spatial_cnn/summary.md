# CIFAR-10 Spatial-Functa CNN PGD Summary

Model: `softlip_spatial_siren_cnn`

Each row is one Full-PGD run of `attacks/full_pgd_cifar10_spatial.py`.

| eps (/255) | n | clean acc | robust acc | robust \| clean | gap (clean-robust) |
|---:|---:|---:|---:|---:|---:|
| 8 | 200 | 0.6250 | 0.0150 | 0.0240 | +0.6100 |
| 16 | 1 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| 16 | 200 | 0.6250 | 0.0150 | 0.0240 | +0.6100 |
| 32 | 200 | 0.6250 | 0.0150 | 0.0240 | +0.6100 |
| 64 | 200 | 0.6250 | 0.0150 | 0.0240 | +0.6100 |
