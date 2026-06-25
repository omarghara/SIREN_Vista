# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_svd_projection/svdproj_readout_scale0.5_e5

Attack mod steps: 3
PGD steps: 100

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| svdproj_readout_scale0.5_e5 | 1 | 100 | 0.8200 | 0.5000 | 0.6098 | +0.3200 |
| svdproj_readout_scale0.5_e5 | 2 | 100 | 0.8200 | 0.3100 | 0.3780 | +0.5100 |
| svdproj_readout_scale0.5_e5 | 4 | 100 | 0.8200 | 0.0700 | 0.0854 | +0.7500 |
| svdproj_readout_scale0.5_e5 | 6 | 100 | 0.8200 | 0.0100 | 0.0122 | +0.8100 |
