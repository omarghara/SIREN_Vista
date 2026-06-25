# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_softlip_warmstart_caps/warmsoftlip_readout_cap50_lam1.0_e5

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warmsoftlip_readout_cap50_lam1.0_e5 | 1 | 200 | 0.7900 | 0.5750 | 0.7278 | +0.2150 |
| warmsoftlip_readout_cap50_lam1.0_e5 | 2 | 200 | 0.7900 | 0.3400 | 0.4304 | +0.4500 |
| warmsoftlip_readout_cap50_lam1.0_e5 | 4 | 200 | 0.7900 | 0.0400 | 0.0506 | +0.7500 |
| warmsoftlip_readout_cap50_lam1.0_e5 | 6 | 200 | 0.7900 | 0.0150 | 0.0190 | +0.7750 |
