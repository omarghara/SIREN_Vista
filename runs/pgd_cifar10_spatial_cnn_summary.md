# CIFAR-10 Spatial-Functa CNN PGD Summary

Each row is one Full-PGD run of attacks/full_pgd_cifar10_spatial.py.

| model | eps (/255) | n | clean acc | robust acc | robust \| clean | gap (clean-robust) |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_spatial_siren_cnn | 1 | 200 | 0.5350 | 0.4950 | 0.6636 | +0.0400 |
| vanilla_spatial_siren_cnn | 2 | 200 | 0.5350 | 0.2700 | 0.4299 | +0.2650 |
| vanilla_spatial_siren_cnn | 4 | 200 | 0.5350 | 0.0950 | 0.1495 | +0.4400 |
| vanilla_spatial_siren_cnn | 6 | 200 | 0.5350 | 0.0350 | 0.0561 | +0.5000 |
| vanilla_spatial_siren_cnn | 8 | 200 | 0.5350 | 0.0200 | 0.0280 | +0.5150 |
| vanilla_spatial_siren_cnn | 16 | 200 | 0.5350 | 0.0050 | 0.0093 | +0.5300 |
