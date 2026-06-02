# CIFAR-10 Spatial-Functa PGD Summary - softlip tiered inner5

Run root: runs/cifar10_spatial_inner5_make5_clfbest_v1/softlip_tiered_e12

Attack mod steps: 5
PGD steps: 200

| model | eps (/255) | n | clean acc | robust acc | robust/clean | gap |
|---|---:|---:|---:|---:|---:|---:|
| softlip_tiered_e12 | 1 | 1000 | 0.7630 | 0.5340 | 0.6999 | +0.2290 |
| softlip_tiered_e12 | 1 | 200 | 0.8200 | 0.5750 | 0.7012 | +0.2450 |
| softlip_tiered_e12 | 2 | 1000 | 0.7630 | 0.3190 | 0.4181 | +0.4440 |
| softlip_tiered_e12 | 2 | 200 | 0.8200 | 0.3150 | 0.3841 | +0.5050 |
| softlip_tiered_e12 | 4 | 1000 | 0.7630 | 0.0640 | 0.0839 | +0.6990 |
| softlip_tiered_e12 | 4 | 200 | 0.8200 | 0.0650 | 0.0793 | +0.7550 |
| softlip_tiered_e12 | 6 | 1000 | 0.7630 | 0.0050 | 0.0066 | +0.7580 |
| softlip_tiered_e12 | 6 | 200 | 0.8200 | 0.0050 | 0.0061 | +0.8150 |
| softlip_tiered_e12 | 8 | 1000 | 0.7630 | 0.0010 | 0.0013 | +0.7620 |
| softlip_tiered_e12 | 8 | 200 | 0.8200 | 0.0000 | 0.0000 | +0.8200 |
