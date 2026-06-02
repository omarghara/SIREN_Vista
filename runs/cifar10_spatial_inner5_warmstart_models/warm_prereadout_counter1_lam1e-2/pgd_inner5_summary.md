# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_warmstart_models/warm_prereadout_counter1_lam1e-2

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warm_prereadout_counter1_lam1e-2 | 1 | 200 | 0.8400 | 0.5350 | 0.6369 | +0.3050 |
| warm_prereadout_counter1_lam1e-2 | 2 | 200 | 0.8400 | 0.2800 | 0.3333 | +0.5600 |
| warm_prereadout_counter1_lam1e-2 | 4 | 200 | 0.8400 | 0.0400 | 0.0476 | +0.8000 |
| warm_prereadout_counter1_lam1e-2 | 6 | 200 | 0.8400 | 0.0000 | 0.0000 | +0.8400 |
| warm_prereadout_counter1_lam1e-2 | 8 | 200 | 0.8400 | 0.0000 | 0.0000 | +0.8400 |
