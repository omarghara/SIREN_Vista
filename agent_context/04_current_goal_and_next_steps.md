# 04 — Current goal and next steps

## Current goal (narrow, on purpose)
**Improve the CIFAR-10 parameter-space classifier's robustness to the PGD attack.**

That's the whole scope right now. We are not trying to make a broad robustness
claim. We are trying to find an intervention that *measurably* raises robust
accuracy under attack, starting with PGD as the screening attack.

## The hard rule for claiming success
**A robustness improvement only counts if it survives stronger attacks.**
PGD-100 at n=100 is a *screen*, not proof. Before calling anything a result:
1. Re-run at larger **n** (>= 500, ideally 1000) so the numbers are outside noise.
2. Increase attack strength: more PGD **steps** and **random restarts**, and more
   inner **mod-steps** (recall `A` grows with inner steps — weak inner fits fake
   robustness).
3. Run an **adaptive / stronger** attack: AutoAttack-style, and an attack that
   properly differentiates (or BPDA-approximates) through the inner fitting loop.
4. Add a **transfer attack** sanity check (perturb against vanilla, apply to the
   defended model) to catch gradient masking.
5. Confirm robust accuracy degrades **smoothly** with eps. A flat-then-cliff curve
   = masking, not robustness.

If a model only looks robust under weak PGD, treat it as **gradient masking** and
discard the claim. (See `context/cursor_context.md` "Strong evaluation rules" and
the Obfuscated Gradients / AutoAttack papers.)

## What NOT to keep doing
sigma_max weight caps (soft-Lipschitz, hard SVD projection) have been tried across
readout / pre-readout / all-sine / modul targets and do not give a clean win. The
math in `03` explains why. Don't re-run more `sigma_max` cap sweeps hoping for a
different outcome — change the lever.

## Recommended next interventions (in priority order)
Ordered by expected payoff / effort, all target the *right* quantity (inverse-map
conditioning), see `03_key_learnings_and_math.md` §4:
1. **Inner-loop latent prior** `lambda || phi ||^2` (Tikhonov). Smallest code change
   with the most direct effect on `1/sigma_min^+`. Sweep lambda; must keep PSNR sane.
2. **Lower omega_0** band-limiting sweep (e.g. 10 -> 6 -> 4) with reconstruction check.
3. **Consistency / adversarial meta-training**: penalize `|| phi(x+delta) - phi(x) ||`
   (or logit consistency) during meta-training.
4. **Jacobian-conditioning penalty** on `J_g` (near-isometry), not just `sigma_max`.

For each: train -> matched makeset -> classifier -> PGD screen -> if promising,
escalate to the stronger-attack protocol above. Always log PSNR and Jacobian
spectrum (`sigma_min^+`, condition number), not just accuracy.

## Useful baselines to compare against
- `warmvanilla_baseline_e5` (matched inner-3 control).
- vanilla e512 inner-5 and softlip-tiered e12 inner-5 (in `context/attack_currenct_results.md`).
Always state the exact checkpoint and clean accuracy in any comparison — the
vanilla-vs-defended ordering has flipped before depending on the baseline.
