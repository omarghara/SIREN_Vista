# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_warmstart_models/warm_readout_cap50_lam1

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warm_readout_cap50_lam1 | 1 | 200 | 0.8300 | 0.5000 | 0.6024 | +0.3300 |
| warm_readout_cap50_lam1 | 2 | 200 | 0.8300 | 0.3100 | 0.3735 | +0.5200 |
| warm_readout_cap50_lam1 | 4 | 200 | 0.8300 | 0.0400 | 0.0482 | +0.7900 |
| warm_readout_cap50_lam1 | 6 | 200 | 0.8300 | 0.0050 | 0.0060 | +0.8250 |
| warm_readout_cap50_lam1 | 8 | 200 | 0.8300 | 0.0050 | 0.0060 | +0.8250 |
