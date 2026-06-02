# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_warmstart_models/warm_orth_lam1e-3

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warm_orth_lam1e-3 | 1 | 200 | 0.7700 | 0.4900 | 0.6299 | +0.2800 |
| warm_orth_lam1e-3 | 2 | 200 | 0.7700 | 0.2600 | 0.3377 | +0.5100 |
| warm_orth_lam1e-3 | 4 | 200 | 0.7700 | 0.0100 | 0.0130 | +0.7600 |
| warm_orth_lam1e-3 | 6 | 200 | 0.7700 | 0.0050 | 0.0065 | +0.7650 |
| warm_orth_lam1e-3 | 8 | 200 | 0.7700 | 0.0000 | 0.0000 | +0.7700 |
