# 03 — Key learnings and the math (what works, what does NOT, and why)

This is the most important file. Read it before proposing a new defense, because
several "obvious" ideas have already been tried and have principled reasons to fail.

## TL;DR
Capping the **largest** singular value (`sigma_max`) of SIREN weights — whether via
soft penalty or hard SVD projection — has **not** improved robustness, and there is
a clear mathematical reason: robustness of the *fitting map* `x -> phi(x)` is
governed by the **smallest** singular value of the decoder Jacobian, not the largest.

## 1. The two maps are different (forward vs inverse)
The SIREN defines the **forward/decoder** map `f_theta: phi -> image`. The
soft-Lipschitz penalty and the SVD projection both bound the forward map:
```text
|| f_theta(phi1) - f_theta(phi2) || <= L || phi1 - phi2 ||   (upper Lipschitz, controlled by sigma_max)
```
But robustness needs the **inverse/fitting** map `image -> phi` to be stable:
```text
|| phi(x+delta) - phi(x) || small
```
A small forward Lipschitz constant does NOT imply a stable inverse. Inverse
stability needs a **lower** bound (bi-Lipschitz / good conditioning):
```text
mu || phi1 - phi2 || <= || f_theta(phi1) - f_theta(phi2) || <= L || phi1 - phi2 ||
```
The missing piece is `mu` (the lower bound). If `mu` is tiny, the decoder has
**flat directions**: `f_theta(phi + v) ≈ f_theta(phi)`, and the fit of `phi`
becomes ill-posed / unstable.

## 2. The Jacobian argument (why sigma_max caps are the wrong lever)
Let `J_g = ∂f_theta/∂phi` be the decoder Jacobian at the fitted point. The inner
fit solves a least-squares problem, so the encoder/fitting Jacobian
`J_E = ∂phi*/∂x` behaves like the **Moore-Penrose pseudoinverse** of `J_g`.
Therefore the input->representation sensitivity scales like:
```text
|| J_E || ~ 1 / sigma_min^+(J_g)      (smallest NON-ZERO singular value of J_g)
```
Consequences:
- Robustness is hurt by **small** `sigma_min^+(J_g)`, i.e. flat directions.
- Capping `sigma_max(W)` lowers `sigma_max(J_g)` (the wrong end). It does nothing
  for `sigma_min^+`, and by shrinking expressivity it can make conditioning
  *worse*, which is exactly what we observed.
- If `phi` is over-parameterized vs the image content, `J_g` has a **null space**.
  Movements of `phi` in the null space cost ~0 reconstruction but can still change
  the classifier output -> unbounded robustness loss that no `sigma_max` cap touches.

## 3. What the experiments actually showed (empirical)
- Soft-Lipschitz: when optimized *alone* it does shrink singular values, so the
  penalty is implemented correctly. But during normal training the reconstruction
  loss **fights** the penalty, so caps don't get enforced cleanly.
- At matched inner steps, the soft-Lipschitz model had **larger** `A(x,delta)`
  (more sensitive) AND worse reconstruction than vanilla. Two losses, no win.
- Hard SVD projection (this session) enforces the cap exactly, but PGD robustness
  is within noise of the matched control (see `02_current_status.md`). Capping
  every sine layer (`all_sine_readout`) also degraded reconstruction. Consistent
  with the math above.
- `A(x,delta)` grows with the number of inner fitting steps (5->200 steps ~8x).
  Short inner loops make models *look* stable because both clean and perturbed
  phi stay near init — a measurement artifact / mild gradient masking, not real
  robustness. This is why matched fitting budgets matter and why stronger attacks
  are mandatory.

## 4. Effective levers the math points to (try these instead)
These target `sigma_min^+` / conditioning / the inverse map directly:
1. **Latent prior in the inner loop** — add `lambda || phi ||^2` (Tikhonov) to the
   inner objective. This lifts the small singular values of the effective system
   `(J_g^T J_g + lambda I)`, directly bounding `|| J_E || <= sigma_max/(sigma_min^2 + lambda)`.
   Cheap, principled, and attacks the right quantity.
2. **Jacobian / isometry penalty** — encourage `J_g` to be near-isometric
   (well-conditioned), e.g. penalize deviation of `J_g^T J_g` from a scaled
   identity, rather than only capping `sigma_max`.
3. **Lower omega_0 (band-limiting)** — sine frequency `omega_0` multiplies every
   layer's effective gain; lowering it shrinks high-frequency directions that
   adversaries exploit. Trade-off: fidelity of high-frequency image content.
4. **Consistency / adversarial meta-training** — directly minimize
   `|| phi(x+delta) - phi(x) ||` (or classifier-logit consistency) during training.
   This optimizes the actual robustness objective instead of a proxy.
5. **Reduce latent over-parameterization** — shrink/structure the phi grid so
   `J_g` has little/no null space (removes the free directions an attacker uses).

## 5. Diagnostics worth running before/after any change
- Per-fit Jacobian spectrum of `J_g = ∂f_theta/∂phi`: `sigma_max`, `sigma_min^+`,
  condition number, `1/sigma_min^+`. Compare variants at **matched reconstruction**.
- `A(x,delta)` for random AND adversarial delta, at matched inner steps.
- Layer amplification `||Δa_l||_2` and ratio `R_l` (`scripts/amplification_analysis.py`).
- Reconstruction PSNR (`scripts/reconstruct_compare.py`) — never let it collapse.

## 6. One-sentence thesis takeaway
> Controlling only the forward upper-Lipschitz constant (sigma_max) of the SIREN
> does not stabilize the fitted modulations; robustness of the bilevel classifier
> is governed by the conditioning / smallest singular value of the decoder
> Jacobian and the null space of phi, so the effective interventions are inner-loop
> latent priors, Jacobian conditioning, band-limiting, and consistency training.
