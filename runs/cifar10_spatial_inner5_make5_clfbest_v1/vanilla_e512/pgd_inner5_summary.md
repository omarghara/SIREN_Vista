# CIFAR-10 Spatial-Functa PGD Summary - vanilla inner5

Run root: runs/cifar10_spatial_inner5_make5_clfbest_v1/vanilla_e512

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_e512 | 1 | 200 | 0.7900 | 0.5400 | 0.6835 | +0.2500 |
| vanilla_e512 | 2 | 200 | 0.7900 | 0.2800 | 0.3544 | +0.5100 |
| vanilla_e512 | 4 | 200 | 0.7900 | 0.0400 | 0.0506 | +0.7500 |
| vanilla_e512 | 6 | 200 | 0.7900 | 0.0000 | 0.0000 | +0.7900 |
| vanilla_e512 | 8 | 200 | 0.7900 | 0.0000 | 0.0000 | +0.7900 |
