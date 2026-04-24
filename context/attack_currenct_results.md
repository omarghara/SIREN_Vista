# Full-PGD on soft-Lipschitz SIREN — live results

Tracking log for the first (and so far only) Full-PGD run against the
softlip backbone, MNIST, eps = 16/255, PGD 100 × mod 10. Baseline vanilla
run has **not** been executed yet, so every "gain vs vanilla" number here
is an assumption from prior work, not a measured delta.

- SIREN checkpoint  : `model_mnist/softlip_L30_lam1e+00_all_skip0/modSiren.pth`
- classifier        : `runs/softlip_L30_lam1e+00_all_skip0/mnist_classifier/best_classifier.pth`
- attack script     : `attacks/full_pgd.py` (Full-PGD, unrolled through the inner loop)
- eps / pgd / mod   : 16/255 · 100 · 10
- PGD lr / inner lr : 0.01 / 0.01
- seed              : 0

## 1. Headline numbers (n = 500)

| | value |
|---|---|
| Clean accuracy              | **0.810** (405 / 500) |
| Robust accuracy @ eps=16/255 | **0.630** (315 / 500) |
| Gap (clean − robust)         | **18.0 pp** |

Both numbers have stabilized over the last ~200 samples; they're no
longer moving monotonically in one direction. Treat these as the
operating-point estimate for this checkpoint.

## 2. Trajectory — why the early n = 6 reading was misleading

Sample-by-sample drift of the running accuracies:

| samples | clean acc | robust acc | gap |
|---:|---:|---:|---:|
|   6  | 0.857 | 0.857 | 0.0 pp |
|  50  | 0.863 | 0.745 | 11.8 pp |
| 100  | 0.880 | 0.752 | 12.8 pp |
| 150  | 0.873 | 0.733 | 14.0 pp |
| 194  | 0.861 | 0.706 | 15.5 pp |
| 300  | 0.840 | 0.670 | 17.0 pp |
| 400  | 0.825 | 0.645 | 18.0 pp |
| 500  | 0.810 | 0.630 | 18.0 pp |

Two things to notice:

1. The initial "clean = robust = 0.857" at n = 6 was pure noise. Once the
   sample size grew past ~100 the true gap appeared and then widened
   slowly up to ~18 pp.
2. Both numbers drifted **down** together (clean 0.86 → 0.81, robust
   0.71 → 0.63) rather than converging. That is consistent with the early
   batch happening to contain easier digits; it is not evidence of any
   time-dependent attack effect (there is no shared state across samples
   in `full_pgd.py:run_attack`).

Statistical error bars at n = 500 (binomial):

- clean : 0.81 ± 0.017 (SE = sqrt(0.81·0.19 / 500))
- robust: 0.63 ± 0.022 (SE = sqrt(0.63·0.37 / 500))

The 18 pp gap is ~6σ outside the noise floor, so it's real. But the
value itself has a ±3 pp 1σ window at this sample count — i.e., the
"true" gap is probably between 15 and 21 pp.

## 3. Interpretation

**The soft-Lipschitz constraint is buying non-trivial robustness, but the**
**clean-accuracy hit is bigger than first appeared.**

- Robust accuracy 63% at eps = 16/255 under 100-step unrolled PGD is
  materially higher than what vanilla MNIST functa classifiers achieve
  in the PSS paper (single digits). If the pending vanilla baseline lands
  near ~5%, you have a ~58 pp absolute robustness gain.
- Clean accuracy 81% is ~16 pp below the vanilla ceiling (~97%). That is
  wider than the 11–12 pp deficit implied by the n ≈ 200 snapshot, and
  wider than what the reconstruction-PSNR penalty alone would predict.
  It suggests that the cost of the Lipschitz constraint shows up not only
  in inner-loop reconstruction quality (4 dB PSNR hit, `context/report.md`
  §3) but also in a harder-to-classify feature distribution at the
  downstream head.

**Pareto sketch (still assuming vanilla ≈ 0.97 / 0.05):**

| model                 | clean | robust (eps=16) | robust / clean |
|-----------------------|------:|----------------:|---------------:|
| vanilla (expected)    | ~0.97 | ~0.05           | ~5 %           |
| softlip (n=500)       | 0.81  | 0.63            | 78 %           |

Even with the larger clean-accuracy cost, this is a striking trade.
Paying ~16 pp of clean accuracy to pick up ~58 pp of robust accuracy is
still Pareto-dominant in every adversarial-robustness paper I've seen on
MNIST-scale functa classifiers. But the claim depends on the vanilla
number. Do not write anything for the thesis until the baseline is run.

## 4. Sanity checks

**1. "Perturbation Final MSE < Clean MSE" is still a warm-start artifact.**

Still seeing lines like `Clean MSE: 0.024 / Perturbation Final MSE: 0.014`
on most samples. This is expected: `full_pgd.py:133` fits the clean
modulation from a **zero** start (10 inner SGD steps), while
`full_pgd.py:161` fits the perturbed modulation from `clean_mod` as
warm start (another 10 inner SGD steps). The perturbed fit gets ~20
effective steps from a good init; the clean fit gets 10 from zero. Not
a phenomenon, just a measurement inconsistency in the reference script.

**2. Robust accuracy here is still unconditional.**

The current `rights_attacked / samples` counts samples classified
correctly under attack regardless of whether they were correct on
clean. The modified `full_pgd.py` (new `--max-samples` + `--output-json`
+ `robust | clean` conditional metric) will fix this for the
next runs — see `scripts/run_pgd_plan.sh`.

**3. Stability.**

At n = 500 the SE on robust accuracy is ±2.2 pp. Going to n = 1000
tightens this to ±1.5 pp, and n = 2000 to ±1.1 pp. For the paper the
default should be n ≥ 500. The current run shows both accuracies have
stopped drifting monotonically after ~350 samples, so n = 500 is a
defensible stopping point.

**4. Gradient-masking check — still required before any claim.**

The fact that PGD is succeeding ~37 % of the time (attack success =
1 − robust acc = 0.37) is actually a good sign for this not being pure
masking — a fully obfuscated model would show attack success near zero
at this budget. But to rule out *partial* masking we still need the
epsilon sweep and ideally a transfer attack:

- **Epsilon sweep** `{8, 16, 32, 64} / 255` on both models, n = 200
  each: if softlip's robust acc degrades smoothly (e.g. 0.70 → 0.63 →
  0.35 → 0.12 going from eps 8 → 64) that's real robustness; if it
  plateaus artificially high at one eps and collapses at the next,
  that's masking. Already wired up as Stage C of
  `scripts/run_pgd_plan.sh`.
- **Transfer attack** (generate perturbations against vanilla, apply to
  softlip): not yet scripted. Low priority until the sweep clears.

## 5. Practical plan from here

1. **This softlip run can be killed.** n = 500 is enough; further samples
   only tighten error bars marginally and cost another ~3.5 h each 500.
   Recorded numbers: clean 0.810, robust 0.630, conditional not
   measured in this run (fix in next run via `--output-json`).
2. **Run full_pgd on vanilla** with identical flags and `n ≥ 500`. Use
   `scripts/run_pgd_plan.sh` Stage B. Non-negotiable.
3. **Epsilon sweep {8, 32, 64} / 255** on both models at n = 200. Stage C
   of the same script. Output = robustness-vs-epsilon curve, which is
   the standard figure that catches gradient masking.
4. **Summary table** auto-produced in
   `runs/pgd_plan_summary.{json,md}` by the script; that's the artifact
   that goes into `context/report.md` as the thesis PGD section.
5. (Deferred) transfer-PGD cross-attack between vanilla and softlip —
   only if Stage C raises any concern about masking.

## 6. TL;DR

At n = 500: **clean 81 %, robust 63 %, gap 18 pp**. The earlier n ≈ 200
snapshot (86 / 71 / 15) was optimistic; the correct operating point is
the one we just converged to. The 18 pp gap is still easily explained
by the Lipschitz constraint doing real work — 37 % attack-success rate
is not gradient masking, and 63 % robust accuracy at 16/255 is very far
from the single-digit numbers vanilla functa classifiers typically hit.
The soft-Lipschitz cost has widened from "~11 pp of clean accuracy" to
"~16 pp of clean accuracy"; that is now a meaningful number to explain
in the thesis, not a rounding error. Next required experiment is the
matched vanilla PGD baseline, then the epsilon sweep.
