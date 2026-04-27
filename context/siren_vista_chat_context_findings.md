# SIREN Vista — Modulation Robustness Context Summary

This document summarizes the main findings, experiments, debugging steps, and conclusions from the chat so far.  
The goal is to preserve context for future work, another AI agent, or a thesis/research notebook.

---

# 1. Project Goal

We are studying a **parameter-space classification pipeline** based on modulated SIRENs.

The main research question is:

> Can we make the full pipeline more robust to small image perturbations by constraining or regularizing the SIREN representation?

The intuitive robustness goal is:

```text
small perturbation in image space
        ↓
small change in fitted modulation vector
        ↓
small change in classifier output
        ↓
same predicted label
```

The classifier does **not** classify the image directly.  
It classifies the **modulation vector** fitted for that image.

---

# 2. Pipeline Reminder

For every image `x`, the pipeline is:

```text
image x
   ↓
fit modulation vector φ(x) using the trained modulated SIREN
   ↓
classifier receives φ(x)
   ↓
predicted class
```

Mathematically:

```text
x → φ(x) → C(φ(x))
```

where:

- `x` is the input image.
- `φ(x)` is the fitted modulation vector for the image.
- `C` is the classifier operating in modulation/parameter space.

For a perturbed image:

```text
x_adv = x + δ
```

the pipeline becomes:

```text
x + δ → φ(x + δ) → C(φ(x + δ))
```

So robustness requires that:

```text
φ(x + δ) ≈ φ(x)
```

and also:

```text
C(φ(x + δ)) ≈ C(φ(x))
```

---

# 3. Important Mathematical Clarification

Initially, the intuition was:

```text
make SIREN linear layers Lipschitz-constrained
        ↓
make the SIREN smoother
        ↓
small image perturbations should not move φ much
```

But this intuition is mathematically incomplete.

The SIREN defines a **forward map**:

```text
Fθ(φ) = reconstructed image
```

So the Lipschitz penalty controls:

```text
φ → reconstructed image
```

For example, an upper Lipschitz bound gives:

```text
||Fθ(φ1) - Fθ(φ2)|| ≤ L ||φ1 - φ2||
```

This means:

> If two modulation vectors are close, their reconstructed images are close.

But what we actually need for robustness is the inverse/fitting direction:

```text
image → fitted modulation vector
```

We want:

```text
||φ(x + δ) - φ(x)|| small
```

when:

```text
||δ|| small
```

These are opposite directions.

---

# 4. Key Insight: Forward Lipschitz Does Not Guarantee Inverse Stability

A forward Lipschitz constraint controls:

```text
large/small changes in φ → changes in image reconstruction
```

But robustness needs:

```text
small changes in image → small changes in φ
```

These are not equivalent.

A function can be very Lipschitz but have an unstable inverse.

Example:

```text
F(φ) = 0
```

This is perfectly `0-Lipschitz`, but it completely collapses all φ values to the same output.  
The inverse is not stable or unique at all.

Therefore:

> Making the SIREN smoother in the forward direction can even make the inverse fitting problem less stable if the representation becomes flatter or less identifiable.

To guarantee inverse stability, we would need something closer to a **bi-Lipschitz** or well-conditioned map:

```text
μ ||φ1 - φ2|| ≤ ||Fθ(φ1) - Fθ(φ2)|| ≤ L ||φ1 - φ2||
```

The lower bound `μ` is the missing piece.

If `μ` is very small, then there are flat directions:

```text
Fθ(φ + v) ≈ Fθ(φ)
```

and fitted modulation vectors can move a lot without changing reconstruction much.

---

# 5. Main Quantity We Decided to Measure

To directly test modulation stability, we defined:

```text
A(x, δ) = ||φ(x + δ) - φ(x)||₂ / ||δ||₂
```

Interpretation:

- Lower `A(x, δ)` means the fitted modulation vector is more stable.
- Higher `A(x, δ)` means small image perturbations move the modulation vector more.

This quantity directly measures the first part of the robustness pipeline:

```text
image perturbation → modulation-vector perturbation
```

---

# 6. Vanilla SIREN Singular-Value Audit

We printed the singular values of the vanilla SIREN.

## 6.1 First SIREN Layer

The first coordinate-input layer is very large:

```text
sine.0 sigma₁ ≈ 4.982
freq = 30
freq * sigma₁ ≈ 149.47
```

This means the first coordinate layer has a very large worst-case amplification from coordinates to hidden representation.

## 6.2 Hidden SIREN Layers

The hidden layers have raw singular values around:

```text
sigma₁ ≈ 0.09 to 0.125
```

But because each SIREN layer uses `freq = 30`, the effective per-layer bound is:

```text
freq * sigma₁ ≈ 2.8 to 3.8
```

So even though the raw singular values look small, the sine frequency makes each hidden layer potentially amplify by about `3×`.

## 6.3 Readout Layer

The readout layer has:

```text
readout sigma₁ ≈ 0.062
```

## 6.4 Modulation Layer

The modulation matrix has:

```text
modul sigma₁ ≈ 1.86
```

This layer maps the modulation vector into the shifts used inside the SIREN.

## 6.5 Worst-Case Bounds

We computed rough worst-case upper bounds:

```text
Lip(coordinate → output) ≈ 2.688e5
Lip(φ → output) ≈ 8.289e4
```

These are loose upper bounds, not actual local sensitivities.  
But they show that the vanilla model is not theoretically small-Lipschitz.

---

# 7. Why `soft_lip_cap = 0.05` Is Extremely Aggressive

The soft-Lipschitz penalty uses a cap `L`.

For sine layers, the effective weight cap is:

```text
sigma(W) ≤ L / freq
```

With:

```text
L = 0.05
freq = 30
```

the target cap becomes:

```text
sigma(W) ≤ 0.05 / 30 ≈ 0.00167
```

But vanilla hidden layers are around:

```text
sigma(W) ≈ 0.09 to 0.125
```

So the penalty is asking the hidden layers to shrink by roughly:

```text
50× to 75×
```

For the modulation layer:

```text
vanilla sigma₁(modul) ≈ 1.86
target cap ≈ 0.00167
```

This is about:

```text
1100× smaller
```

## Conclusion

`L = 0.05` is not a mild regularizer.  
It is an extremely strong constraint that forces the model into a very different scale regime.

More realistic starting values are:

```text
L = 4
L = 3
L = 2
L = 1
L = 0.5
```

because vanilla hidden layers already have:

```text
freq * sigma(W) ≈ 2.8 to 3.8
```

---

# 8. Debugging the Soft-Lipschitz Penalty

We created a debug training cell that prints, for each layer:

```text
||g_mse||       gradient norm from reconstruction loss
||g_pen||       gradient norm from Lipschitz penalty
||g_train||     gradient norm from the actual training loss
||ΔW||          actual optimizer update size
relative ΔW     ||ΔW|| / ||W||
sigma₁ before
sigma₁ after
top-k singular values before and after
```

This was used to understand why the soft-Lipschitz penalty was not converging well.

---

# 9. Total-Loss Debug Result

When training with:

```text
total_loss = reconstruction MSE + soft-Lipschitz penalty
```

we observed:

```text
Penalty loss is huge,
but singular values do not consistently decrease.
```

Some singular values even increased.

This happened especially when using:

```text
outer_lr = 1e-4
```

because the reconstruction gradients often dominated the penalty gradients.

Example qualitative pattern:

```text
hidden-layer ||g_mse|| >> hidden-layer ||g_pen||
```

So the reconstruction objective pushes the model to preserve expressive/high-gain weights, while the penalty pushes spectral norms down.

## Conclusion

The optimization has a real conflict:

```text
reconstruction wants expressive / high-gain weights
Lipschitz penalty wants lower-gain weights
```

This explains why the penalty loss did not cleanly converge during normal training.

---

# 10. Penalty-Only Debug Result

We then ran the same debug loop with:

```text
train_loss = penalty_loss only
```

using:

```text
DEBUG_START_FROM = vanilla
DEBUG_LOSS_MODE = penalty_only
DEBUG_OUTER_LR = 1e-4
DEBUG_SOFT_LIP_CAP = 0.05
```

## Result

The singular values decreased consistently in most layers.

Examples:

```text
sine.0 : 4.9824 → 4.9736
sine.1 : 0.0927 → 0.0913
sine.6 : 0.1057 → 0.0927
sine.7 : 0.1198 → 0.0922
sine.8 : 0.1248 → 0.0944
sine.9 : 0.1253 → 0.0918
readout: 0.0620 → 0.0549
```

The modulation layer barely changed:

```text
modul: 1.8620 → 1.8615
```

## Conclusion

The soft-Lipschitz penalty implementation is working.

The problem is not that the penalty is mathematically broken.

The problem is that during normal training:

```text
MSE reconstruction gradients fight against the spectral penalty.
```

---

# 11. Issue With the Modulation Layer

The modulation layer is large:

```text
modul shape = 2560 × 512
```

Even when the penalty gradient is large, its top singular value barely decreases.

This suggests that the modulation layer may require special treatment.

Possible future directions:

```text
1. stronger lambda for the modulation layer
2. separate layer-specific lambdas
3. direct spectral normalization
4. projection-based spectral control
5. different treatment of the modulation matrix
```

---

# 12. Layer-Specific Lambda Insight

A single global `lambda` is probably not ideal.

Observed pattern:

```text
hidden-layer penalty gradients ≈ 0.17–0.24
modulation-layer penalty gradient ≈ 3.5
```

So the same penalty weight affects different layer types very differently.

A better design may be:

```text
lambda_sine    = 10 or 50
lambda_modul   = 1
lambda_readout = 1
lambda_first   = 1
```

because hidden-layer MSE gradients can become much larger than hidden-layer penalty gradients.

---

# 13. Empirical Vanilla Perturbation Amplification — Initial Test

We measured:

```text
A(x, δ) = ||φ(x + δ) - φ(x)||₂ / ||δ||₂
```

for the vanilla model using random image perturbations.

With only:

```text
5 inner fitting steps
```

we got:

```text
mean_A ≈ 0.0037 to 0.0042
median_A ≈ 0.0034 to 0.0035
```

This initially suggested:

```text
random small image perturbation
→ very small movement in modulation space
```

But this was only with 5 inner steps.

---

# 14. Important Finding: Sensitivity Increases With Inner-Loop Steps

We then tested vanilla with:

```text
5, 20, 50, 100, 200 inner fitting steps
```

For `eps = 8/255`, the vanilla `mean_A` was:

```text
5 steps    → 0.00409
20 steps   → 0.00840
50 steps   → 0.01464
100 steps  → 0.02242
200 steps  → 0.03319
```

So from 5 to 200 inner steps:

```text
A increased by about 8×
```

## Interpretation

With only 5 steps, both:

```text
φ(x)
φ(x + δ)
```

remain close to the initialization, so they are naturally close to each other.

With more inner optimization steps, the modulation vector better fits the image, and the clean and perturbed images lead to more separated fitted modulation vectors.

## Reconstruction Tradeoff

As inner steps increased, reconstruction improved:

```text
5 steps    MSE ≈ 0.02694
20 steps   MSE ≈ 0.02110
50 steps   MSE ≈ 0.01883
100 steps  MSE ≈ 0.01748
200 steps  MSE ≈ 0.01646
```

So we observed a tradeoff:

```text
better reconstruction
        ↔
more sensitivity in fitted modulation vectors
```

This is a major result.

---

# 15. Vanilla vs Soft-Lipschitz: A(x, δ) Comparison

We compared:

```text
vanilla
vs
soft-Lipschitz
```

using the same `A(x, δ)` metric across:

```text
fit_steps = 5, 20, 50, 100, 200
eps = 1/255, 2/255, 4/255, 8/255, 16/255
```

## Main Finding

The current soft-Lipschitz model usually has **larger A** than vanilla.

That means:

```text
soft-Lipschitz caused larger movement in φ under the same image perturbation
```

This is the opposite of the desired effect.

---

# 16. Example: eps = 8/255

For `eps = 8/255`:

```text
5 steps:
vanilla  mean_A = 0.00409
softlip  mean_A = 0.00460

20 steps:
vanilla  mean_A = 0.00840
softlip  mean_A = 0.01270

50 steps:
vanilla  mean_A = 0.01464
softlip  mean_A = 0.02053

100 steps:
vanilla  mean_A = 0.02242
softlip  mean_A = 0.03114

200 steps:
vanilla  mean_A = 0.03319
softlip  mean_A = 0.04698
```

Soft-Lipschitz consistently produced larger modulation sensitivity.

---

# 17. Soft-Lipschitz Also Reconstructs Worse

For clean reconstruction MSE:

```text
5 steps:
vanilla  MSE = 0.02694
softlip  MSE = 0.03769

20 steps:
vanilla  MSE = 0.02110
softlip  MSE = 0.02842

50 steps:
vanilla  MSE = 0.01883
softlip  MSE = 0.02387

100 steps:
vanilla  MSE = 0.01748
softlip  MSE = 0.02134

200 steps:
vanilla  MSE = 0.01646
softlip  MSE = 0.01922
```

So the current soft-Lipschitz model has:

```text
worse reconstruction
and
larger modulation movement
```

This means the current soft-Lipschitz implementation/training setting is not improving robustness.

---

# 18. Interpretation of the Soft-Lipschitz Failure

The result matches the earlier mathematical concern.

The soft-Lipschitz penalty controls the forward map:

```text
φ → reconstructed image
```

But we care about the inverse/fitting map:

```text
image → fitted φ
```

If the forward map becomes flatter or less expressive, then the optimizer may need to move `φ` more to explain the same image difference.

In simple terms:

```text
if changing φ affects the image less,
then to match a changed image,
the optimizer may need to change φ more.
```

So the current soft-Lipschitz constraint may make the inverse fitting problem worse.

---

# 19. Major Research Insight So Far

The original hypothesis:

```text
smaller spectral norms → more stable modulation vectors
```

is not automatically true.

A better statement is:

```text
controlling only the forward Lipschitz constant of the SIREN is insufficient;
we must also consider the conditioning/invertibility of the φ → image map.
```

This is probably the most important conceptual conclusion so far.

---

# 20. What We Should Measure Next

The next important object is the Jacobian:

```text
J = ∂Fθ(φ) / ∂φ
```

where:

```text
Fθ(φ) = reconstructed image
```

To understand inverse stability, we should measure:

```text
σmax(J)
σmin(J)
condition number = σmax(J) / σmin(J)
```

Especially important:

```text
σmin(J)
```

If `σmin(J)` is very small, the map has flat directions and the fitted modulation vector can become unstable.

If soft-Lipschitz decreases `σmax` but decreases `σmin` even more, it can worsen conditioning and make the inverse map less stable.

---

# 21. Suggested Next Diagnostic

Compare vanilla and soft-Lipschitz on:

```text
J = ∂reconstructed image / ∂φ
```

for fitted modulation vectors.

For each model and image, compute:

```text
model
image index
fit steps
reconstruction MSE
σmax(J)
σmin(J)
condition number
1 / σmin(J)
```

Expected explanation if current results continue:

```text
softlip may have smaller forward gain,
but worse conditioning,
which explains why φ(x + δ) moves more.
```

---

# 22. Proposed Future Directions

## 22.1 Use More Realistic Lipschitz Caps

Instead of starting from:

```text
L = 0.05
```

try:

```text
L = 4
L = 3
L = 2
L = 1
L = 0.5
```

This is more aligned with the vanilla scale:

```text
freq * sigma(W) ≈ 2.8 to 3.8
```

## 22.2 Use a Gradual Schedule

Instead of enforcing a strict cap from the start:

```text
epoch 0–5:    L = 4
epoch 5–10:   L = 3
epoch 10–15:  L = 2
epoch 15–20:  L = 1
```

This lets the model first learn reconstruction, then gradually regularize spectral norms.

## 22.3 Use Layer-Specific Penalties

Use different lambdas for different layer groups:

```text
lambda_sine
lambda_first
lambda_readout
lambda_modul
```

because one global lambda does not balance the gradients well.

## 22.4 Directly Regularize Modulation Stability

A more directly aligned regularizer would be:

```text
R = ||φ(x + δ) - φ(x)||²
```

This directly penalizes the thing we care about.

Downside:

```text
requires fitting φ twice per sample
```

so it is more expensive.

## 22.5 Regularize the Classifier in Modulation Space

Even if φ moves, the classifier should not be too sensitive.

A possible classifier-side regularizer:

```text
||∇φ C(φ)|| small
```

or adversarial training directly in modulation space.

## 22.6 Measure Adversarial Rather Than Random Perturbations

So far, the `A(x, δ)` experiments used random perturbations.

Need to test adversarial perturbations:

```text
maximize ||φ(x + δ) - φ(x)|| 
or maximize classifier loss
subject to ||δ||∞ ≤ ε
```

Random noise stability does not imply adversarial robustness.

---

# 23. Current Main Conclusions

## Conclusion 1

The pipeline robustness depends on modulation stability:

```text
small image perturbation
→ small fitted modulation change
→ stable classifier output
```

## Conclusion 2

A forward Lipschitz penalty on the SIREN does not automatically stabilize fitted modulation vectors.

## Conclusion 3

The current soft-Lipschitz penalty implementation works mechanically: penalty-only training reduces spectral norms.

## Conclusion 4

During full training, reconstruction loss fights against the spectral penalty.

## Conclusion 5

The current cap `L = 0.05` is extremely aggressive and likely too strict.

## Conclusion 6

Vanilla modulation sensitivity increases with the number of inner fitting steps.

## Conclusion 7

The current soft-Lipschitz model gives worse reconstruction and larger modulation sensitivity than vanilla.

## Conclusion 8

The next key concept is conditioning of the map:

```text
φ → reconstructed image
```

not just its upper Lipschitz constant.

## Conclusion 9

The next experiment should measure the singular values of:

```text
J = ∂Fθ(φ) / ∂φ
```

and compare vanilla vs soft-Lipschitz.

---

# 24. Thesis-Level Interpretation

A good thesis-level statement based on the current experiments:

> A soft upper-Lipschitz constraint on the SIREN does not necessarily stabilize fitted modulation vectors. Although the penalty successfully reduces spectral norms when optimized alone, during full training it conflicts with the reconstruction objective. Empirically, the current soft-Lipschitz model produces worse reconstruction and larger modulation displacement under random image perturbations than the vanilla model. This suggests that controlling only the forward Lipschitz constant of the SIREN is insufficient; the conditioning and inverse stability of the modulation fitting process must also be considered.

---

# 25. Recommended Immediate Next Steps

1. Compute Jacobian conditioning:

```text
J = ∂Fθ(φ) / ∂φ
```

for vanilla and soft-Lipschitz.

2. Compare:

```text
σmax(J)
σmin(J)
condition number
```

3. Train new soft-Lipschitz variants with more realistic caps:

```text
L = 4, 3, 2, 1
```

4. Rerun `A(x, δ)` comparison.

5. Compare at equal reconstruction quality, not only equal number of inner steps.

6. Eventually test adversarial perturbations, not only random perturbations.

---

# 26. Important Warning for Future Agents

Do not assume:

```text
lower spectral norms = better robustness
```

The experiments so far suggest the opposite may happen if the inverse fitting map becomes worse conditioned.

The correct research direction is:

```text
robustness requires stable image → φ fitting
```

and this depends on both:

```text
forward smoothness
and
inverse conditioning / identifiability
```

