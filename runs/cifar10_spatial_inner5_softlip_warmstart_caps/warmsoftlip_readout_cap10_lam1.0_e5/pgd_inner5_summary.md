# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_softlip_warmstart_caps/warmsoftlip_readout_cap10_lam1.0_e5

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warmsoftlip_readout_cap10_lam1.0_e5 | 1 | 200 | 0.7950 | 0.5150 | 0.6478 | +0.2800 |
| warmsoftlip_readout_cap10_lam1.0_e5 | 2 | 200 | 0.7950 | 0.3350 | 0.4214 | +0.4600 |
| warmsoftlip_readout_cap10_lam1.0_e5 | 4 | 200 | 0.7950 | 0.0650 | 0.0818 | +0.7300 |
| warmsoftlip_readout_cap10_lam1.0_e5 | 6 | 200 | 0.7950 | 0.0100 | 0.0126 | +0.7850 |
