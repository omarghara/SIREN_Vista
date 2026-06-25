# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_svd_projection/svdproj_all_sine_readout_scale0.5_e5

Attack mod steps: 3
PGD steps: 100

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| svdproj_all_sine_readout_scale0.5_e5 | 1 | 100 | 0.7700 | 0.5700 | 0.7143 | +0.2000 |
| svdproj_all_sine_readout_scale0.5_e5 | 2 | 100 | 0.7700 | 0.3300 | 0.4286 | +0.4400 |
| svdproj_all_sine_readout_scale0.5_e5 | 4 | 100 | 0.7700 | 0.0800 | 0.1039 | +0.6900 |
| svdproj_all_sine_readout_scale0.5_e5 | 6 | 100 | 0.7700 | 0.0200 | 0.0260 | +0.7500 |
