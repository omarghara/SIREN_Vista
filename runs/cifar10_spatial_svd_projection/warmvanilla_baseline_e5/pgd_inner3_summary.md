# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_svd_projection/warmvanilla_baseline_e5

Attack mod steps: 3
PGD steps: 100

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warmvanilla_baseline_e5 | 1 | 100 | 0.8300 | 0.5800 | 0.6988 | +0.2500 |
| warmvanilla_baseline_e5 | 2 | 100 | 0.8300 | 0.3300 | 0.3976 | +0.5000 |
| warmvanilla_baseline_e5 | 4 | 100 | 0.8300 | 0.0400 | 0.0482 | +0.7900 |
| warmvanilla_baseline_e5 | 6 | 100 | 0.8300 | 0.0100 | 0.0120 | +0.8200 |
