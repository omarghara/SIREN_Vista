# CIFAR-10 Softlip-Warmstart Cap Experiment

- target: `pre_readout`
- scale relative to softlip checkpoint: `0.10`
- spectral-cap lambda: `1.0`
- train epochs: `5`
- warm-start checkpoint: `model_cifar10/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cifar_spatial_tiered_lam1e-02_sine_and_readout_train50000_test10000/modSiren.pth`
- cap-reference checkpoint: `model_cifar10/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e12_inner3_moptsgd_adamphi3_lr1e-02_softlip_cifar_spatial_tiered_lam1e-02_sine_and_readout_train50000_test10000/modSiren.pth`
- trained checkpoint: `model_cifar10/cifar10_spatial_warmsoftlip_prereadout_cap10_lam1.0_e5/modSiren.pth`
- run root: `runs/cifar10_spatial_inner5_softlip_warmstart_caps/warmsoftlip_prereadout_cap10_lam1.0_e5`
- makeset inner iterations: `5`
- classifier epochs: `40`
- PGD eps list: `1 2 4 6`
- PGD samples: `200`
- PGD steps: `200`
- PGD inner phi steps: `5`
