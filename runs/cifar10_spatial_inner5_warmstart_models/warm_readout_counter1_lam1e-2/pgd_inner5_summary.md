# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_warmstart_models/warm_readout_counter1_lam1e-2

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warm_readout_counter1_lam1e-2 | 1 | 200 | 0.7200 | 0.5150 | 0.7153 | +0.2050 |
| warm_readout_counter1_lam1e-2 | 2 | 200 | 0.7200 | 0.3950 | 0.5486 | +0.3250 |
| warm_readout_counter1_lam1e-2 | 4 | 200 | 0.7200 | 0.0700 | 0.0972 | +0.6500 |
| warm_readout_counter1_lam1e-2 | 6 | 200 | 0.7200 | 0.0250 | 0.0347 | +0.6950 |
