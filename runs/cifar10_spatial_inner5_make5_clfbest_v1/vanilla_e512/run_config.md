# CIFAR-10 Vanilla Spatial-Functa Inner-5 Run Config

- SIREN checkpoint: `model_cifar10/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/modSiren.pth`
- functaset train_all: `runs/cifar10_spatial_inner5_make5_clfbest_v1/vanilla_e512/functaset/vanilla_e512_inner5_train_all50000.pkl`
- functaset test: `runs/cifar10_spatial_inner5_make5_clfbest_v1/vanilla_e512/functaset/vanilla_e512_inner5_test.pkl`
- makeset inner steps: `5`
- makeset inner lr: `0.01`
- makeset optimizer: `sgd`
- classifier type: `cnn`
- classifier lr: `0.003`
- classifier width: `256`
- classifier dropout: `0.1`
- classifier normalize phi: `1`
- classifier epochs: `40`
- classifier checkpoint: `runs/cifar10_spatial_inner5_make5_clfbest_v1/vanilla_e512/cifar10_cnn_classifier_best_sweep_inner5/best_classifier.pth`
- PGD mod steps: `5`
- PGD steps: `200`
- PGD LR: `0.01`
- PGD eps list: `1 2 4 6 8`
- PGD max samples: `200`
