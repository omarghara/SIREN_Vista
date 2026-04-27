# Hardcap90 soft-Lipschitz experiment — conclusion

This note summarizes the experiment run with the modified
`SIREN_Vista/variants/soft_lipschitz.py` that hard-codes per-layer spectral
norm caps at **90% of the measured vanilla spectral norm**.

The goal was not to impose a global `L / freq` cap. The goal was milder and
more diagnostic:

> Can we slightly shrink the SIREN backbone's spectral norms layer-by-layer
> while preserving reconstruction quality and making fitted modulations less
> sensitive to small input perturbations?

---

## 1. Experiment setup

### Code variant

`SIREN_Vista/variants/soft_lipschitz.py` currently contains a hardcoded
`_collect_layers(...)` for this experiment.

It ignores the global `L` for the SIREN layers and instead uses raw spectral
norm caps:

```text
cap(layer) = 0.9 * sigma_vanilla(layer)
```

Penalized layers:

- `sine.0`
- `sine.1` ... `sine.9`
- `readout` / `hidden2rgb`

Not penalized:

- `modul`

The run used:

```text
checkpoint  : SIREN_Vista/model_mnist/softlip_L1_lam1e+00_sine_and_readout/modSiren.pth
run outputs : SIREN_Vista/runs/softlip_hardcap90_lam1e+00_sine_and_readout/
variant     : soft_lipschitz
lambda      : 1.0
apply_to    : sine_and_readout
skip_first  : false
epochs      : best checkpoint after epoch index 4
```

The checkpoint path has `softlip_L1...` because it was trained before
`trainer.py --model-name` was added. The downstream artifacts were saved under
the intended descriptive name `softlip_hardcap90_lam1e+00_sine_and_readout`.

---

## 2. Did the 10% spectral cap work?

Mostly yes for the SIREN sine layers, but not perfectly. Offline spectral audit
of the saved checkpoint:

| layer | final sigma | hardcap90 cap | final / cap | final / vanilla |
|---|---:|---:|---:|---:|
| `sine.0` | 4.544292 | 4.484198 | 1.013 | 0.912 |
| `sine.1` | 0.089495 | 0.083397 | 1.073 | 0.966 |
| `sine.2` | 0.090273 | 0.085370 | 1.057 | 0.952 |
| `sine.3` | 0.088053 | 0.083698 | 1.052 | 0.947 |
| `sine.4` | 0.089151 | 0.083986 | 1.061 | 0.955 |
| `sine.5` | 0.092345 | 0.087737 | 1.053 | 0.947 |
| `sine.6` | 0.099359 | 0.095142 | 1.044 | 0.940 |
| `sine.7` | 0.111169 | 0.107863 | 1.031 | 0.928 |
| `sine.8` | 0.113890 | 0.112355 | 1.014 | 0.912 |
| `sine.9` | 0.111465 | 0.112778 | 0.988 | 0.890 |
| `readout` | 0.062662 | 0.055793 | 1.123 | 1.011 |
| `modul` | 3.788900 | not penalized | - | - |

Interpretation:

- The sine layers were reduced relative to vanilla by roughly **3% to 11%**.
- The deepest sine layers (`sine.8`, `sine.9`) got closest to the intended 10%
  reduction.
- Most sine layers remain slightly **above** their 90% cap, so the penalty did
  not fully enforce the target in the best saved checkpoint.
- `readout` did **not** shrink; it ended slightly larger than vanilla. The
  readout cap is too weak or the penalty is being outweighed by MSE pressure.
- `modul` grew substantially and was intentionally not constrained. This is
  important: the forward SIREN became slightly smaller, but the
  modulation-to-shift map may compensate by becoming larger.

**Conclusion:** the hardcap90 loss partially achieved its intended spectral
effect on the sine stack, but it did not produce a clean hard constraint.

---

## 3. Reconstruction quality

From:

```text
SIREN_Vista/runs/softlip_hardcap90_lam1e+00_sine_and_readout/reconstruction_eval.json
```

### Test split

| inner iters | MSE | PSNR | SSIM |
|---:|---:|---:|---:|
| 5 | 0.01252 | 19.72 | 0.779 |
| 20 | 0.00686 | 22.48 | 0.834 |
| 50 | 0.00378 | 25.08 | 0.866 |
| 100 | 0.00231 | 27.21 | 0.884 |
| 200 | 0.00144 | 29.27 | 0.899 |

For reference, the earlier vanilla reconstruction report had test PSNR:

| inner iters | vanilla PSNR | hardcap90 PSNR | difference |
|---:|---:|---:|---:|
| 5 | 20.70 | 19.72 | -0.99 dB |
| 20 | 24.58 | 22.48 | -2.11 dB |
| 50 | 26.31 | 25.08 | -1.23 dB |
| 100 | 27.35 | 27.21 | -0.14 dB |
| 200 | 28.24 | 29.27 | +1.03 dB |

Important caveat: the vanilla reference here is the older reconstruction eval
from `runs/vanilla/reconstruction_eval.json`, not necessarily the fully trained
`vanilla_e40` checkpoint. Still, the hardcap90 result is clearly much less
destructive than the earlier `softlip_L30_lam1e+00_all_skip0` run, which had
roughly a **4 dB** PSNR deficit at the classifier operating point.

**Conclusion:** hardcap90 preserves reconstruction quality far better than the
previous `apply_to=all` soft-Lipschitz run. At `iters=5`, the cost is about
**1 dB** rather than ~4 dB. At high inner iterations it reaches vanilla-like or
better reconstruction on this comparison.

---

## 4. Perturbation sensitivity from the diagnostics notebook

Notebook:

```text
SIREN_Vista/notebooks/model_diagnostics copy.ipynb
```

The notebook compared the empirical amplification quantity `A(x, delta)` for
vanilla vs hardcap90 across fitting budgets and small noise scales. Lower is
better in this diagnostic: it means the fitted modulation / representation
moves less for the same image-space perturbation.

Representative mean `A(x, delta)` values:

| fit steps | epsilon (/255) | vanilla mean A | hardcap90 mean A |
|---:|---:|---:|---:|
| 5 | 16 | 0.00373 | 0.00266 |
| 20 | 16 | 0.00826 | 0.00531 |
| 50 | 16 | 0.01382 | 0.00974 |
| 100 | 16 | 0.02010 | 0.01510 |
| 200 | 16 | 0.02796 | 0.02274 |

Across the notebook table, hardcap90 is consistently lower than vanilla at the
same fit-step and epsilon settings.

**Conclusion:** this variant appears to reduce local perturbation
amplification in the fitted representation. This is the most encouraging
diagnostic result of the experiment.

---

## 5. What the experiment does and does not prove

### What it supports

- Mild per-layer spectral caps are feasible: they do not destroy
  reconstruction the way tighter global caps did.
- The sine stack can be nudged downward in spectral norm by a few percent to
  ~10%.
- The diagnostic perturbation metric improves: fitted representations are less
  sensitive under the tested random perturbations.

### What it does not prove yet

- It does **not** prove PGD robustness. Full-PGD has not yet been run for
  `softlip_hardcap90_lam1e+00_sine_and_readout`.
- It does **not** prove a certificate. The final sigmas still exceed several
  caps, and `modul` is unconstrained.
- It does **not** isolate whether the improvement comes from the sine caps, the
  changed training trajectory, or compensation through `modul`.

---

## 6. Main interpretation

This is a better direction than the original soft-Lipschitz setting.

The first `apply_to=all` soft-Lipschitz experiment mostly constrained `modul`
and paid a large reconstruction cost. In contrast, hardcap90 constrains the
actual SIREN weights, avoids the severe reconstruction collapse, and gives a
cleaner diagnostic story:

```text
slightly smaller SIREN spectra
        -> similar reconstruction quality
        -> lower empirical perturbation amplification
```

The major warning is that `modul` grew to sigma about **3.79** while being
unpenalized. That means the system may be compensating for reduced SIREN
weights by making the phi-to-shift map stronger. This is not necessarily bad
for classification, but it weakens any claim that the full phi-to-output map is
more Lipschitz.

---

## 7. Recommended next steps

1. **Run Full-PGD for hardcap90** using `scripts/run_pgd_single_model.sh` or a
   small custom command:
   - start with epsilon 16/255, `n=200` or `n=500`,
   - then sweep 8/32/64 if the result is promising.
2. **Audit `modul` explicitly** in the thesis narrative. Its growth is likely
   an important compensation mechanism.
3. **Try one follow-up variant:** hardcap90 on SIREN weights **plus a mild
   modul penalty**, not the very tight `L/freq` modul penalty from earlier.
4. **Compare against fully trained vanilla e40 reconstruction**, not only the
   older `runs/vanilla/reconstruction_eval.json`, if reconstruction quality is
   going into the final thesis.
5. **Keep this as a thesis-worthy diagnostic result even if PGD does not win:**
   it demonstrates that spectral shaping can reduce local representation
   sensitivity without catastrophic fidelity loss.

---

## 8. One-line conclusion

The hardcap90 experiment is a **successful diagnostic variant**: it partially
shrinks SIREN spectral norms, preserves reconstruction quality much better than
the earlier soft-Lipschitz run, and reduces empirical perturbation
amplification in the notebook; the remaining open question is whether this
translates into Full-PGD robustness once the classifier is attacked.
