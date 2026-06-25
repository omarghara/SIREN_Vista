# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_svd_projection/svdproj_modul_scale0.5_e5

Attack mod steps: 3
PGD steps: 100

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| svdproj_modul_scale0.5_e5 | 1 | 100 | 0.8300 | 0.5200 | 0.6265 | +0.3100 |
| svdproj_modul_scale0.5_e5 | 2 | 100 | 0.8300 | 0.3000 | 0.3614 | +0.5300 |
| svdproj_modul_scale0.5_e5 | 4 | 100 | 0.8300 | 0.0700 | 0.0843 | +0.7600 |
| svdproj_modul_scale0.5_e5 | 6 | 100 | 0.8300 | 0.0100 | 0.0120 | +0.8200 |
