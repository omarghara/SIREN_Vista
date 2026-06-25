# PGD robustness plan — report (vanilla e40 vs softlip)

**Vanilla baseline:** `vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256`  
(SIREN: `model_mnist/vanilla_e40_.../modSiren.pth`; classifier: `runs/vanilla_e40_.../mnist_classifier/best_classifier.pth`.)  

**Softlip:** `softlip_L30_lam1e+00_all_skip0` (unchanged from prior plan).  

**Source artifacts:** `pgd_plan_summary.md`, `pgd_plan_summary.json` under `SIREN_Vista/runs/`.  

**Attack:** `attacks/full_pgd.py` — Full PGD, 100 outer steps, outer LR 0.01, inner modulation steps 10, inner LR 0.01; L∞ constraint ε/255.

---

## Setup

| Setting | Value |
|--------|--------|
| PGD iterations | 100 |
| PGD outer LR | 0.01 |
| Inner mod steps | 10, inner LR 0.01 |
| ε sweep | 8, 16, 32, 64 (in **/255** units) |
| n samples | **200** for ε ∈ {8, 32, 64}; **500** for ε = 16 |

---

## Summary table (from `pgd_plan_summary.md`)

| model   | eps (/255) |   n | clean acc | robust acc | robust \| clean | gap (clean−robust) |
|---------|-----------:|----:|----------:|-----------:|-----------------:|-------------------:|
| softlip |          8 | 200 |    0.9750 |     0.9150 |           0.9385 |            +0.0600 |
| vanilla |          8 | 200 |    0.9900 |     0.9500 |           0.9596 |            +0.0400 |
| softlip |         16 | 500 |    0.9680 |     0.6780 |           0.6983 |            +0.2900 |
| vanilla |         16 | 500 |    0.9720 |     0.7120 |           0.7325 |            +0.2600 |
| softlip |         32 | 200 |    0.9750 |     0.3400 |           0.3487 |            +0.6350 |
| vanilla |         32 | 200 |    0.9900 |     0.2150 |           0.2172 |            +0.7750 |
| softlip |         64 | 200 |    0.9750 |     0.2650 |           0.2718 |            +0.7100 |
| vanilla |         64 | 200 |    0.9900 |     0.0300 |           0.0303 |            +0.9600 |

---

## Findings

### Clean accuracy

On this evaluation, **vanilla (e40)** reaches **~99%** clean accuracy on the attacked subsets (198/200 at ε 8/32/64; 486/500 at ε 16), while **softlip** stays near **~97.5%** (195/200; 484/500). So the comparison is **not** at matched clean error; vanilla is a stronger *standard* baseline on clean accuracy here.

### Who wins on robust accuracy?

| ε (/255) | n | Winner (robust acc) | softlip | vanilla (e40) | Margin (van − soft) |
|---------:|--:|---------------------|--------:|----------------:|---------------------:|
| 8 | 200 | **vanilla** | 0.915 | **0.950** | +0.035 |
| 16 | 500 | **vanilla** | 0.678 | **0.712** | +0.034 |
| 32 | 200 | **softlip** | **0.340** | 0.215 | −0.125 |
| 64 | 200 | **softlip** | **0.265** | 0.030 | −0.235 |

- **Low perturbation (ε = 8, 16):** the trained **vanilla e40** classifier is **more robust** than softlip under this Full-PGD budget, while also having higher clean accuracy. That is a meaningful change versus a weaker vanilla baseline (e.g. older ~91% clean runs): a stronger vanilla both cleans up and stays ahead at moderate ε.

- **High perturbation (ε = 32, 64):** **softlip** retains substantially higher robust accuracy. Vanilla collapses toward chance-level robustness at ε = 64 (**3%** vs softlip **26.5%**).

### Gap (clean − robust)

Vanilla’s gap is **smaller** at ε = 8 (+4.0 pp vs +6.0 pp for softlip) because vanilla’s robust accuracy is higher. At ε = 32 both show a **+0.775** vs **+0.635** gap in the table; read this together with absolute robust numbers: vanilla’s larger gap coexists with **lower** robust acc at 32 (0.215 vs 0.340). At ε = 64 vanilla’s gap is **+0.96** with only **3%** robust — almost all clean predictions become attackable.

### Conditional robustness (robust \| clean)

Vanilla leads at ε 8 and 16 (e.g. **0.9596** vs **0.9385** at ε 8). At ε 32 and 64, softlip’s conditional rates (**0.3487**, **0.2718**) dominate vanilla’s (**0.2172**, **0.0303**), consistent with the unconditional story.

---

## Interpretation (short)

With a **strong vanilla** (40-epoch README-style training, bundled under `vanilla_e40_...`), **parameter-space adversarial robustness under Full-PGD is competitive or better than softlip at small–medium L∞ budgets** on this protocol, but **softlip still provides a large advantage under strong attacks** (ε ≥ 32/255 on this setup). Any paper or slide comparing “vanilla vs softlip” should name the exact vanilla checkpoint and clean accuracy, since the conclusion **flips at low ε** when vanilla is properly trained.

---

## Artifact paths

| Kind | Path |
|------|------|
| Summary MD | `/workspace/SIREN_Vista/runs/pgd_plan_summary.md` |
| Summary JSON | `/workspace/SIREN_Vista/runs/pgd_plan_summary.json` |
| Softlip per-run | `/workspace/SIREN_Vista/runs/softlip_L30_lam1e+00_all_skip0/pgd_plan/eps*_n*.{json,log}` |
| Vanilla e40 per-run | `/workspace/SIREN_Vista/runs/vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256/pgd_plan/eps*_n*.{json,log}` |

---

## Limitations

1. **Unequal n:** ε = 16 uses 500 samples; others use 200.  
2. **Matched-risk / matched-accuracy:** not enforced; vanilla is cleaner and more robust at low ε here.  
3. **Single attack recipe:** other budgets or AutoAttack may change ordering.  
4. **Subset evaluation:** same protocol as `run_pgd_plan.sh` (first n samples per run).

---

*Report reflects the completed PGD plan summarized in `pgd_plan_summary.*` after switching vanilla to `vanilla_e40_lr0.01_cw512_md512_do0p2_cd3_bs256`.*
