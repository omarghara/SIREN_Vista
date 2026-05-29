# CIFAR-10 Spatial-Functa CNN PGD Summary

Model: `softlip_spatial_siren_cnn_smalleps`

Each row is one Full-PGD run of `attacks/full_pgd_cifar10_spatial.py`.

| eps (/255) | n | clean acc | robust acc | robust \| clean | gap (clean-robust) |
|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 0.6250 | 0.2150 | 0.3360 | +0.4100 |
| 2 | 200 | 0.6250 | 0.1400 | 0.2160 | +0.4850 |
| 4 | 200 | 0.6250 | 0.0350 | 0.0560 | +0.5900 |
| 6 | 200 | 0.6250 | 0.0150 | 0.0240 | +0.6100 |
