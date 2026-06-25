# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_softlip_warmstart_caps/warmsoftlip_prereadout_cap50_lam1.0_e5

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warmsoftlip_prereadout_cap50_lam1.0_e5 | 1 | 200 | 0.8050 | 0.5300 | 0.6584 | +0.2750 |
| warmsoftlip_prereadout_cap50_lam1.0_e5 | 2 | 200 | 0.8050 | 0.3450 | 0.4286 | +0.4600 |
| warmsoftlip_prereadout_cap50_lam1.0_e5 | 4 | 200 | 0.8050 | 0.0450 | 0.0559 | +0.7600 |
| warmsoftlip_prereadout_cap50_lam1.0_e5 | 6 | 200 | 0.8050 | 0.0000 | 0.0000 | +0.8050 |
