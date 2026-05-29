# CIFAR-10 Spatial-Functa CNN PGD Summary

Model: `vanilla_spatial_siren_cnn_smalleps`

Each row is one Full-PGD run of `attacks/full_pgd_cifar10_spatial.py`.

| eps (/255) | n | clean acc | robust acc | robust \| clean | gap (clean-robust) |
|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 0.7550 | 0.4000 | 0.5298 | +0.3550 |
| 2 | 200 | 0.7550 | 0.1050 | 0.1391 | +0.6500 |
| 4 | 200 | 0.7550 | 0.0050 | 0.0066 | +0.7500 |
| 6 | 200 | 0.7550 | 0.0050 | 0.0066 | +0.7500 |
