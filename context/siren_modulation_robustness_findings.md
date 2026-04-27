# Summary of Findings and Conclusions — SIREN / Modulation Robustness Debugging

## 1. Project pipeline reminder

The pipeline we are studying is:

```text
image x
   ↓
fit modulation vector φ(x) using the trained modulated SIREN
   ↓
classifier receives φ(x)
   ↓
predicted class
```

So the classifier does **not** classify the image directly.

It classifies the **modulation vector**:

```text
C(φ(x))
```

For a perturbed image:

```text
x_adv = x + δ
```

the pipeline becomes:

```text
x + δ
   ↓
fit φ(x + δ)
   ↓
C(φ(x + δ))
```

Therefore, for robustness, we want:

```text
small image perturbation δ
   ↓
small modulation change φ(x + δ) - φ(x)
   ↓
small classifier-output change
   ↓
same predicted label
```

The key quantity we started measuring is:

```text
A(x, δ) = ||φ(x + δ) - φ(x)||₂ / ||δ||₂
```

Lower `A(x, δ)` means the fitted modulation vector is more stable under image perturbations.

---

## 2. Important mathematical clarification

At first, the intuition was:

```text
make the SIREN linear layers Lipschitz-constrained
   ↓
make the whole SIREN smoother
   ↓
small image perturbations should not move φ much
```

But this intuition is incomplete.

The SIREN defines a forward map:

```text
Fθ(φ) = reconstructed image
```

So the Lipschitz constraint controls:

```text
φ → image reconstruction
```

Meaning:

```text
||Fθ(φ₁) - Fθ(φ₂)|| ≤ L ||φ₁ - φ₂||
```

But the pipeline needs stability of the inverse/fitting map:

```text
image → fitted φ
```

Meaning we want:

```text
||φ(x + δ) - φ(x)|| small
```

These are opposite directions.

A forward Lipschitz bound does **not** automatically imply inverse stability.

### Key conclusion

Controlling only the upper Lipschitz constant of the SIREN does **not** guarantee that fitted modulation vectors will be stable.

To guarantee inverse stability, we would need something closer to a **bi-Lipschitz** or well-conditioned map:

```text
μ ||φ₁ - φ₂|| ≤ ||Fθ(φ₁) - Fθ(φ₂)|| ≤ L ||φ₁ - φ₂||
```

The missing part is the lower bound `μ`.

If `μ` is very small, then the map has flat directions:

```text
Fθ(φ + v) ≈ Fθ(φ)
```

and the fitted modulation vector may become unstable.

---

## 3. Vanilla singular-value audit

We printed the singular values of the vanilla SIREN.

### Main observations

The first coordinate-input SIREN layer is very large:

```text
sine.0 sigma₁ ≈ 4.98
freq = 30
freq * sigma₁ ≈ 149.47
```

The hidden layers have raw singular values around:

```text
sigma₁ ≈ 0.09 to 0.125
```

But since each sine layer uses `freq = 30`, the effective per-layer upper bound is:

```text
freq * sigma₁ ≈ 2.8 to 3.8
```

The modulation layer has:

```text
modul sigma₁ ≈ 1.86
```

The readout layer has:

```text
readout sigma₁ ≈ 0.062
```

### Worst-case upper bounds

The coordinate-input worst-case bound was approximately:

```text
Lip(x_coord → output) ≤ 2.688e5
```

The modulation-vector worst-case bound was approximately:

```text
Lip(φ → output) ≤ 8.289e4
```

These are only loose worst-case bounds, but they show that the vanilla model is not theoretically small-Lipschitz.

---

## 4. Why `soft_lip_cap = 0.05` is extremely aggressive

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

the target becomes:

```text
sigma(W) ≤ 0.05 / 30 ≈ 0.00167
```

But vanilla hidden layers have:

```text
sigma(W) ≈ 0.09 to 0.125
```

So we are asking the hidden layers to shrink by approximately:

```text
50× to 75×
```

For the modulation layer:

```text
vanilla sigma₁(modul) ≈ 1.86
target cap ≈ 0.00167
```

That is approximately:

```text
1100× smaller
```

### Conclusion

`L = 0.05` is not a mild regularization setting.

It is an extremely strong constraint that forces the model into a very different scale regime.

A more realistic starting range is:

```text
L = 4
L = 3
L = 2
L = 1
L = 0.5
```

because vanilla hidden layers have:

```text
freq * sigma(W) ≈ 2.8 to 3.8
```

---

## 5. Debugging the soft-Lipschitz penalty

We created a debug training cell that prints, per layer:

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

The goal was to understand why the soft-Lipschitz penalty was not converging well.

---

## 6. Finding: with total loss, MSE fights the penalty

When training with:

```text
total_loss = reconstruction MSE + soft-Lipschitz penalty
```

we observed:

```text
Penalty loss is huge
but singular values do not consistently decrease
```

Some singular values even increased.

This happened especially with larger learning rate:

```text
outer_lr = 1e-4
```

because reconstruction gradients became dominant in many layers.

Example pattern:

```text
hidden layers:
||g_mse|| can be much larger than ||g_pen||
```

So the model tries to preserve or improve reconstruction, while the penalty tries to shrink the spectral norms.

### Conclusion

The optimization problem has a real conflict:

```text
reconstruction wants expressive / higher-gain weights
Lipschitz penalty wants lower-gain weights
```

This explains why the penalty loss did not cleanly converge during normal training.

---

## 7. Penalty-only debug result

We then ran:

```text
train_loss = penalty_loss only
```

with:

```text
DEBUG_LOSS_MODE = "penalty_only"
DEBUG_OUTER_LR = 1e-4
DEBUG_SOFT_LIP_CAP = 0.05
DEBUG_START_FROM = "vanilla"
```

### Result

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

### Conclusion

The penalty implementation itself is working.

The problem is **not** that the penalty is mathematically broken.

The problem is that, during normal training, the reconstruction objective fights against the penalty.

---

## 8. Issue with the modulation layer

The modulation layer is huge:

```text
modul shape = 2560 × 512
```

Even when its penalty gradient is large, its top singular value barely decreases.

This suggests that controlling the modulation layer may require special treatment.

Possible future ideas:

```text
1. stronger lambda for modul
2. separate lambda per layer type
3. direct spectral normalization / projection
4. different treatment for the modulation matrix
```

---

## 9. Layer-specific lambda insight

A single global `lambda` is probably not ideal.

We observed:

```text
hidden-layer penalty gradients ≈ 0.17–0.24
modulation-layer penalty gradient ≈ 3.5
```

So the same penalty weight affects different layers very differently.

A better design may use layer-specific penalty weights:

```text
lambda_sine    = 10 or 50
lambda_modul   = 1
lambda_readout = 1
lambda_first   = 1
```

This is because hidden-layer MSE gradients can become much larger than hidden-layer penalty gradients.

---

## 10. Empirical vanilla perturbation amplification

We measured:

```text
A(x, δ) = ||φ(x + δ) - φ(x)||₂ / ||δ||₂
```

for the vanilla model using random image perturbations.

Initially, with only 5 inner optimization steps, `A` was very small:

```text
mean_A ≈ 0.0037 to 0.0042
median_A ≈ 0.0034 to 0.0035
```

This means:

```text
random small image perturbation
→ very small movement in modulation space
```

But this was with only 5 inner steps.

---

## 11. Important finding: sensitivity increases with inner-loop steps

We then tested vanilla with different numbers of inner fitting steps:

```text
5, 20, 50, 100, 200
```

For `eps = 8/255`, the vanilla `mean_A` increased as follows:

```text
5 steps    → 0.00409
20 steps   → 0.00840
50 steps   → 0.01464
100 steps  → 0.02242
200 steps  → 0.03319
```

So from 5 to 200 inner steps:

```text
A increases by about 8×
```

At the same time, reconstruction MSE improved:

```text
5 steps    → 0.02694
20 steps   → 0.02110
50 steps   → 0.01883
100 steps  → 0.01748
200 steps  → 0.01646
```

### Conclusion

There is a clear tradeoff:

```text
more accurate reconstruction
↔ more sensitive modulation representation
```

Short inner-loop fitting can make the modulation vector look artificially stable, because both clean and perturbed images remain close to the initialization.

As the inner loop fits better, small perturbations cause larger movement in modulation space.

---

## 12. Vanilla vs soft-Lipschitz A(x, δ) comparison

We compared:

```text
vanilla vs softlip
```

using the same quantity:

```text
A(x, δ) = ||φ(x + δ) - φ(x)||₂ / ||δ||₂
```

### Main finding

The current soft-Lipschitz model usually has **larger** `A` than vanilla.

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

So the current soft-Lipschitz model makes the fitted modulation vector **more sensitive**, not less sensitive.

---

## 13. Soft-Lipschitz also reconstructs worse

The soft-Lipschitz model also has worse clean reconstruction MSE.

Examples:

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

### Conclusion

The current soft-Lipschitz model is doing two bad things:

```text
worse reconstruction
and
larger modulation movement under perturbation
```

Therefore, the current soft-Lipschitz method is **not yet a successful robustness method**.

---

## 14. Why soft-Lipschitz can make A worse

This result matches the mathematical concern.

The penalty controls the forward map:

```text
φ → reconstructed image
```

But we care about the inverse/fitting map:

```text
image → fitted φ
```

If the forward map becomes flatter or less expressive, then changing `φ` affects the reconstructed image less.

That can force the optimizer to move `φ` more in order to explain the same image perturbation.

In simple terms:

```text
If changing φ affects the image less,
then to match a changed image,
the optimizer may need to change φ more.
```

So reducing the forward Lipschitz constant can make the inverse map worse if the representation becomes poorly conditioned.

---

## 15. Main thesis-level insight so far

The naive hypothesis:

```text
lower spectral norms → more stable modulations
```

is not automatically true.

A better and more accurate hypothesis is:

```text
Spectral regularization may affect the stability of fitted modulation vectors,
but this depends on the conditioning of the φ → image map,
not only on its upper Lipschitz constant.
```

The current experiments suggest:

```text
controlling only σmax / upper Lipschitz is insufficient
```

We also need to understand:

```text
σmin
conditioning
inverse stability
flat directions in modulation space
```

---

## 16. Recommended next diagnostic

The next important object to measure is the Jacobian:

```text
J = ∂Fθ(φ) / ∂φ
```

where:

```text
Fθ(φ) = reconstructed image
```

For each fitted modulation vector, compare vanilla vs soft-Lipschitz:

```text
σmax(J)
σmin(J)
condition number = σmax(J) / σmin(J)
effective inverse scale ≈ 1 / σmin(J)
```

If soft-Lipschitz has smaller `σmax(J)` but much smaller `σmin(J)`, then it is worse-conditioned.

That would explain why `A(x, δ)` becomes larger.

---

## 17. Key conclusion to write in the thesis/report

A concise thesis-style summary:

> A soft upper-Lipschitz constraint on the SIREN does not necessarily stabilize the fitted modulation vectors. Although the penalty successfully reduces spectral norms when optimized alone, during normal training the reconstruction objective fights against the penalty. Empirically, the current soft-Lipschitz model produces worse reconstruction quality and larger modulation displacement under random input perturbations compared to vanilla. This suggests that controlling only the forward Lipschitz constant of the SIREN is insufficient; the conditioning of the inverse fitting map from image to modulation must also be considered.

---

## 18. Practical next steps

The next experiments should be:

```text
1. Measure Jacobian conditioning: ∂Fθ(φ) / ∂φ
2. Compare σmax, σmin, and condition number for vanilla vs softlip
3. Try more realistic caps: L = 4, 3, 2, 1, 0.5
4. Try layer-specific lambdas
5. Compare A(x, δ) at matched reconstruction quality, not only matched inner steps
6. Eventually test adversarial perturbations, not only random perturbations
```

The most important comparison is:

```text
Can we reduce A(x, δ) without significantly worsening reconstruction MSE?
```

That should become the central experimental question.
