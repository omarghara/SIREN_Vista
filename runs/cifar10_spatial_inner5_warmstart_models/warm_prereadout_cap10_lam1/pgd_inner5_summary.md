# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_warmstart_models/warm_prereadout_cap10_lam1

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warm_prereadout_cap10_lam1 | 1 | 200 | 0.8150 | 0.5450 | 0.6687 | +0.2700 |
| warm_prereadout_cap10_lam1 | 2 | 200 | 0.8150 | 0.3450 | 0.4233 | +0.4700 |
| warm_prereadout_cap10_lam1 | 4 | 200 | 0.8150 | 0.0200 | 0.0245 | +0.7950 |
| warm_prereadout_cap10_lam1 | 6 | 200 | 0.8150 | 0.0000 | 0.0000 | +0.8150 |
| warm_prereadout_cap10_lam1 | 8 | 200 | 0.8150 | 0.0000 | 0.0000 | +0.8150 |
