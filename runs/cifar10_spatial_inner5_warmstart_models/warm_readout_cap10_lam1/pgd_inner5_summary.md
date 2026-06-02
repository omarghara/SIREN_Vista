# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_warmstart_models/warm_readout_cap10_lam1

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warm_readout_cap10_lam1 | 1 | 200 | 0.7900 | 0.5300 | 0.6709 | +0.2600 |
| warm_readout_cap10_lam1 | 2 | 200 | 0.7900 | 0.3200 | 0.4051 | +0.4700 |
| warm_readout_cap10_lam1 | 4 | 200 | 0.7900 | 0.0400 | 0.0506 | +0.7500 |
| warm_readout_cap10_lam1 | 6 | 200 | 0.7900 | 0.0000 | 0.0000 | +0.7900 |
| warm_readout_cap10_lam1 | 8 | 200 | 0.7900 | 0.0000 | 0.0000 | +0.7900 |
