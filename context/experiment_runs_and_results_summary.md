# Experiment runs, backbones, and classifier results

Generated as an inventory of what exists under `model_cifar10/`, `runs/`, and classifier checkpoints. Paths are relative to the repo root `SIREN_Vista/` unless stated otherwise.

## How artifacts line up

- **Backbone (INR meta-network)**: `model_cifar10/<slug>/modSiren.pth`. The directory name `<slug>` is the canonical experiment id.
- **Run directory**: `runs/<slug>/` — may contain `functaset/*.pkl`, classifier subdirs, and sometimes `run_summary.md`.
- **Classifier training** reads functa pickles only; it does not load the INR at train time. The **same slug** ties a backbone on disk to the run folder used for functasets and classifiers.
- **Best classifier metrics** below come from `torch.load(..., map_location="cpu")` on each `best_classifier.pth` (`epoch`, `accuracy`).

---

## CIFAR-10 — classifier results (`best_classifier.pth`)

| Run slug (prefix) | Classifier dir | Best top-1 (%) | Best epoch |
| --- | --- | --- | --- |
| `functa_like_cifar10_spatial_paper_finer_..._train50000_test10000` | `cifar10_classifier` | 49.28 | 116 |
| `functa_like_cifar10_spatial_paper_siren_..._e512_..._train50000_test10000` | `cifar10_classifier` | 49.10 | 86 |
| same | `cifar10_cnn_classifier` | 70.45 | 71 |
| same | `cifar10_vit_train50k_test10k` | 56.43 | 77 |
| `functa_like_cifar10_spatial_paper_siren_..._e5_..._train50000_test10000` | `cifar10_classifier` | 43.67 | 80 |
| `functa_like_cifar10_spatial_paper_finer_..._train5000_test1000` | `cifar10_classifier` | 37.60 | 89 |
| `functa_like_cifar10_spatial_finer_h512_..._train5000_test1000` | `cifar10_classifier` | 46.10 | 79 |
| `functa_like_cifar10_spatial_siren_h512_..._train5000_test1000` | `cifar10_classifier` | 40.80 | 25 |
| `functa_like_cifar10_finer_h512_md1024_d10_..._train5000_test1000` | `cifar10_classifier` | 43.00 | 81 |
| `functa_like_cifar10_finer_h512_md512_d10_..._train5000_test1000` | `cifar10_classifier` | 41.90 | 53 |
| `functa_like_cifar10_finer_h512_md1024_d15_..._train5000_test1000` | `cifar10_classifier` | 39.70 | 74 |
| `fourier_siren_cifar10_h256_md512_d10_nf64_sig10_e5_make20` | `cifar10_classifier` | 45.67 | 38 |
| `vanilla_cifar10` | `cifar10_classifier` | 47.21 | 32 |

**Spatial paper 50k/10k takeaway:** FINER MLP **49.28%**; SIREN trained **512** outer epochs MLP **49.10%**; same functaset slug with **ViT** **56.43%** and **small CNN** **70.45%** (different heads, same underlying functa features). SIREN with only **5** outer epochs on the same schedule slug reaches **43.67%** MLP — functaset for that run exists (`train_all50000.pkl` / `test.pkl` under `runs/.../functaset/`).

---

## MNIST — classifier results (`best_classifier.pth`)

| Run slug | Best top-1 (%) | Best epoch |
| --- | --- | --- |
| `softlip_first95_rest80_lam1e+00_sine_and_readout` | 98.12 | 22 |
| `softlip_hardcap90_lam1e+00_sine_and_readout` | 98.19 | 26 |
| `softlip_L0.5_lam1e+00_all_skip0` | 97.99 | 36 |
| `softlip_L30_lam1e+00_all_skip0` | 97.89 | 23 |
| `vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256` | 97.78 | 34 |
| `softlip_hardcap85_lam1e+00_sine_and_readout` | 97.10 | 36 |
| `softlip_hardcap90_modul125_lam1e+00_all` | 97.08 | 33 |
| `softlip_hardcap75_lam1e+00_sine_and_readout` | 96.26 | 34 |
| `softlip_L30_lam1e+00_all_skip0_v1` | 96.69 | 39 |

---

## CIFAR-10 — backbone checkpoints (`model_cifar10/*/modSiren.pth`)

All entries below have `modSiren.pth`. Grouped by family; full directory name equals the **slug** used under `runs/<slug>/` when that run exists.

### Spatial — paper-style (h256, lat 8×16, freq 10, `moptsgd`, `adamphi3`)

| Slug fragment | `inr_type` / notes | Outer epochs (from name) | `checkpoint_summary.md` |
| --- | --- | --- | --- |
| `...spatial_paper_finer_..._train50000_test10000` | finer | e5 | yes |
| `...spatial_paper_finer_..._train5000_test1000` | finer | e5 | yes |
| `...spatial_paper_siren_..._e512_..._train50000_test10000` | siren | e512 | yes |
| `...spatial_paper_siren_..._e5_..._train50000_test10000` | siren | e5 | yes |
| `...spatial_paper_siren_..._train5000_test1000` | siren | e5 | no |

**Recorded backbone metrics** (from `checkpoint_summary.md`):

- **Paper FINER 50k/10k**: best epoch 4, best mean outer loss ≈ **0.005627**, requested epochs 5, `inr_type`: finer.
- **Paper SIREN 50k/10k e512**: best epoch 54, best mean outer loss ≈ **0.003208**, requested epochs 512, `inr_type`: siren, `is_spatial`: True.
- **Paper SIREN 50k/10k e5**: best epoch 4, best mean outer loss ≈ **0.015503**, requested epochs 5, `inr_type`: siren.

### Spatial — h512 exploratory (`spatial_finer` / `spatial_siren`, `adamphi200`, 5k/1k)

- `functa_like_cifar10_spatial_finer_h512_md1024_d10_lat8x16_..._train5000_test1000`
- `functa_like_cifar10_spatial_finer_h512_md512_d10_lat8x8_..._train5000_test1000`
- `functa_like_cifar10_spatial_finer_h512_md512_d8_lat8x8_..._train5000_test1000`
- `functa_like_cifar10_spatial_siren_h512_md1024_d10_lat8x16_..._train5000_test1000`

### Non-spatial functa-like — FINER / SIREN / Fourier

- FINER: `functa_like_cifar10_finer_h512_md1024_d10_..._train5000_test1000`, `..._d15_..._bias{1,2,4}_..._make50`, `functa_like_cifar10_finer_h512_md512_d10_..._train5000_test1000`
- SIREN: `functa_like_cifar10_siren_h512_md1024_d15_..._make50`, `functa_like_cifar10_siren_h512_md128_d15_..._make50`
- Fourier: `functa_like_cifar10_fourier_h512_md1024_d15_nf64_sig5_rawxy_norm01_..._make50` and `..._w060_...`, `..._w090_...`, `..._w0120_...`

### Other CIFAR backbone

- `fourier_siren_cifar10_h256_md1024_d10_nf64_sig10_inclin_norm01_e7_make50` (under `model_cifar10/`; run with classifier uses a different Fourier-SIREN slug under `runs/` — see classifier table)

---

## Run folders (`runs/`) — functaset vs classifier

| Slug | `functaset/` present | `best_classifier.pth` present |
| --- | --- | --- |
| `functa_like_cifar10_finer_*_train5000_test1000` (d10 / md512 / d15 train split) | yes (where makeset was run) | yes for the three in classifier table |
| `functa_like_cifar10_spatial_*` paper and h512 | yes for most; paper FINER 5k dir had no `functaset/` at last scan | yes where listed above |
| `functa_like_cifar10_spatial_finer_h512_md512_d10_lat8x8_...` | yes | no |
| `functa_like_cifar10_finer_*_make50` (no `_train5000_test1000`) | mixed | no |
| `functa_like_cifar10_fourier_*`, `functa_like_cifar10_siren_*` | some yes | no |
| `fourier_siren_cifar10_h256_md512_*` | no | yes (classifier only) |
| `vanilla_cifar10` | no | yes |
| `vanilla`, `non_pgd_three_models`, `softlip_L0.05_*` | no | no |
| SoftLIP / vanilla MNIST runs | no | yes for most softlip + `vanilla_e40_*` |

**Note:** The spatial paper FINER **5k** classifier exists, but this scan did not find a `functaset/` directory under that run path (pickles may live elsewhere or were removed after training).

---

## Summary files on disk

- **Backbone** `checkpoint_summary.md`: only under the four **spatial paper** CIFAR dirs listed above (FINER 50k/5k, SIREN e512 50k, SIREN e5 50k).
- **Run** `run_summary.md`: under spatial paper FINER (50k and 5k slugs) and spatial paper SIREN **e512** 50k slug; links backbone path + default `cifar10_classifier` result.
- **Classifier** `checkpoint_summary.md`: under some `runs/.../cifar10_classifier/` for spatial paper FINER and SIREN (e512) 50k runs.

---

## Repro / naming reminders

- Spatial **SIREN vs FINER** is determined by the backbone / functaset slug (`spatial_paper_siren` vs `spatial_paper_finer`), not by the classifier script.
- Functa filenames follow `{slug}_train_all{N}.pkl`, `{slug}_test.pkl`, etc.; there is no generic `cifar10_train.pkl` for these pipelines.
- For ViT/CNN on the same spatial SIREN e512 50k functaset, save dirs are siblings: `cifar10_classifier`, `cifar10_vit_train50k_test10k`, `cifar10_cnn_classifier`.

To refresh numbers after new training, re-scan with:

`find runs -name best_classifier.pth` and load each checkpoint’s `accuracy` / `epoch` as above.

### Full classifier paths (copy-paste)

- `runs/functa_like_cifar10_spatial_paper_finer_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e5_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/cifar10_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/cifar10_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/cifar10_cnn_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e512_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/cifar10_vit_train50k_test10k/best_classifier.pth`
- `runs/functa_like_cifar10_spatial_paper_siren_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e5_inner3_moptsgd_adamphi3_lr1e-02_train50000_test10000/cifar10_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_spatial_paper_finer_h256_md1024_d6_lat8x16_freq10.0_nearest_lc_norm01_extlr3e-05_e5_inner3_moptsgd_adamphi3_lr1e-02_train5000_test1000/cifar10_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_spatial_finer_h512_md1024_d10_lat8x16_nearest_lc_norm01_extlr1e-05_e5_inner3_adamphi200_lr3e-03_train5000_test1000/cifar10_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_spatial_siren_h512_md1024_d10_lat8x16_nearest_lc_norm01_extlr1e-05_e5_inner3_adamphi200_lr3e-03_train5000_test1000/cifar10_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_finer_h512_md1024_d10_freq30.0_bias2_scaledetach_norm01_extlr1e-05_e5_inner3_adamphi200_lr3e-03_train5000_test1000/cifar10_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_finer_h512_md512_d10_freq30.0_bias2_scaledetach_norm01_extlr1e-05_e5_inner3_adamphi200_lr3e-03_train5000_test1000/cifar10_classifier/best_classifier.pth`
- `runs/functa_like_cifar10_finer_h512_md1024_d15_freq30.0_bias2_scaledetach_norm01_extlr1e-05_e10_inner3_make50_adamphi200_lr3e-03_train5000_test1000/cifar10_classifier/best_classifier.pth`
- `runs/fourier_siren_cifar10_h256_md512_d10_nf64_sig10_e5_make20/cifar10_classifier/best_classifier.pth`
- `runs/vanilla_cifar10/cifar10_classifier/best_classifier.pth`
- `runs/softlip_first95_rest80_lam1e+00_sine_and_readout/mnist_classifier/best_classifier.pth`
- `runs/softlip_hardcap90_lam1e+00_sine_and_readout/mnist_classifier/best_classifier.pth`
- `runs/softlip_L0.5_lam1e+00_all_skip0/mnist_classifier/best_classifier.pth`
- `runs/softlip_L30_lam1e+00_all_skip0/mnist_classifier/best_classifier.pth`
- `runs/vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256/mnist_classifier/best_classifier.pth`
- `runs/softlip_hardcap85_lam1e+00_sine_and_readout/mnist_classifier/best_classifier.pth`
- `runs/softlip_hardcap90_modul125_lam1e+00_all/mnist_classifier/best_classifier.pth`
- `runs/softlip_hardcap75_lam1e+00_sine_and_readout/mnist_classifier/best_classifier.pth`
- `runs/softlip_L30_lam1e+00_all_skip0_v1/mnist_classifier/best_classifier.pth`
