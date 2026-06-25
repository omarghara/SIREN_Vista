# CIFAR-10 Spatial-Functa PGD Summary - inner5

Run root: runs/cifar10_spatial_inner5_softlip_warmstart_caps/warmsoftlip_prereadout_cap10_lam1.0_e5

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| warmsoftlip_prereadout_cap10_lam1.0_e5 | 1 | 200 | 0.8150 | 0.5350 | 0.6564 | +0.2800 |
| warmsoftlip_prereadout_cap10_lam1.0_e5 | 2 | 200 | 0.8150 | 0.3200 | 0.3926 | +0.4950 |
| warmsoftlip_prereadout_cap10_lam1.0_e5 | 4 | 200 | 0.8150 | 0.0250 | 0.0307 | +0.7900 |
| warmsoftlip_prereadout_cap10_lam1.0_e5 | 6 | 200 | 0.8150 | 0.0050 | 0.0061 | +0.8100 |
