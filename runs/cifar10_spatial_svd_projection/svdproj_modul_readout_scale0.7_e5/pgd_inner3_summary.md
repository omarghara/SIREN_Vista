# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_svd_projection/svdproj_modul_readout_scale0.7_e5

Attack mod steps: 3
PGD steps: 100

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| svdproj_modul_readout_scale0.7_e5 | 1 | 100 | 0.7800 | 0.5300 | 0.6795 | +0.2500 |
| svdproj_modul_readout_scale0.7_e5 | 2 | 100 | 0.7800 | 0.3200 | 0.4103 | +0.4600 |
| svdproj_modul_readout_scale0.7_e5 | 4 | 100 | 0.7800 | 0.0500 | 0.0641 | +0.7300 |
| svdproj_modul_readout_scale0.7_e5 | 6 | 100 | 0.7800 | 0.0100 | 0.0128 | +0.7700 |
