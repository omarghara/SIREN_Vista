# Experiment Report: Hardcap90 Per-Layer Soft-Lipschitz SIREN

## 1. Goal

The goal of this experiment was to improve the robustness of the SIREN modulation-classification pipeline.

The pipeline is:

```text
image x
   ↓
fit modulation vector φ(x)
   ↓
classify φ(x)
```

The robustness objective is:

```text
small image perturbation δ
   ↓
small change in fitted modulation φ(x + δ) − φ(x)
   ↓
stable classifier prediction
```

The main stability quantity we measured was:

```text
A(x, δ) = ||φ(x + δ) − φ(x)||₂ / ||δ||₂
```

Lower `A(x, δ)` means the fitted modulation vector is less sensitive to image perturbations.

---

## 2. Motivation From Earlier Findings

Earlier experiments showed that using a very aggressive global soft-Lipschitz cap, such as `L = 0.05`, was too strong.

The problems with the earlier global cap were:

```text
1. It severely compressed the SIREN spectral norms.
2. It worsened reconstruction quality.
3. It increased modulation sensitivity A(x, δ).
4. It did not improve the desired image → modulation stability.
```

The key mathematical insight was:

```text
A forward Lipschitz constraint on φ → image does not automatically guarantee
stability of the inverse/fitting map image → φ.
```

So instead of applying an extreme global cap, we designed a milder, per-layer cap calibrated to the vanilla model.

---

## 3. New Experiment: Hardcap90

The new experiment was called:

```text
softlip_hardcap90_lam1e+00_sine_and_readout
```

The checkpoint was saved at:

```text
model_mnist/softlip_L1_lam1e+00_sine_and_readout/modSiren.pth
```

The classifier was saved at:

```text
runs/softlip_hardcap90_lam1e+00_sine_and_readout/mnist_classifier/best_classifier.pth
```

The idea was:

```text
For each SIREN layer, set the spectral cap to 90% of its vanilla singular value.
```

This means we aimed to reduce the layer spectral bounds by about:

```text
10%
```

The goal was not to crush the network, but to mildly regularize each layer.

---

## 4. What Was Penalized

For this run, we penalized only the SIREN reconstruction weights:

```text
sine.0
sine.1
...
sine.9
readout / hidden2rgb
```

We did **not** penalize:

```text
modul
```

The run used:

```text
APPLY_TO = sine_and_readout
lambda   = 1
EXT_LR   = 5e-5
INT_LR   = 0.01
```

The soft-Lipschitz cap argument was technically:

```text
soft_lip_cap = 1.0
```

but in this hardcoded experiment, the actual caps were hardcoded in `variants/soft_lipschitz.py` as 90% of the measured vanilla singular values.

---

## 5. Training Observations

During training, the soft-Lipschitz penalty decreased smoothly.

Approximate observed penalty trend:

```text
epoch 0 avg pen ≈ 0.1537
epoch 1 avg pen ≈ 0.0761
epoch 2 avg pen ≈ 0.0357
epoch 3 avg pen ≈ 0.0155
epoch 4 loss     ≈ 0.0255
```

The largest SIREN singular value moved toward the intended 90% cap.

For `sine.0`:

```text
vanilla sigma₁ ≈ 4.982
90% cap        ≈ 4.484
during training it moved down close to this range
```

The run showed that the hardcap90 penalty was feasible and did not destabilize training.

### Important Observation: Modulation Matrix Compensation

Although `modul` was not penalized, its spectral norm increased during training:

```text
modul started around ≈ 1.84
modul grew above ≈ 3.7 during training
```

This suggests the model partly compensated for reduced SIREN spectral norms by increasing the modulation mapping.

This is important for interpretation:

```text
Constraining only the SIREN weights can cause the unconstrained modulation matrix to grow.
```

However, despite this growth, the final results were positive.

---

## 6. Classifier Result

The downstream classifier trained on the hardcap90 functaset achieved:

```text
Best Accuracy: 98.19%
```

This shows that the new modulation representation remained highly classifiable.

---

## 7. Reconstruction Quality

The hardcap90 model achieved strong reconstruction quality.

### Train Reconstruction

| Inner iters | MSE | PSNR mean/median | SSIM mean |
|---:|---:|---:|---:|
| 5 | 1.237e-02 | 19.76 / 19.36 | 0.7855 |
| 20 | 6.869e-03 | 22.45 / 22.14 | 0.8372 |
| 50 | 3.817e-03 | 25.01 / 24.79 | 0.8673 |
| 100 | 2.361e-03 | 27.09 / 26.98 | 0.8847 |
| 200 | 1.496e-03 | 29.09 / 29.05 | 0.8993 |

### Test Reconstruction

| Inner iters | MSE | PSNR mean/median | SSIM mean |
|---:|---:|---:|---:|
| 5 | 1.252e-02 | 19.72 / 19.32 | 0.7789 |
| 20 | 6.859e-03 | 22.48 / 22.14 | 0.8343 |
| 50 | 3.780e-03 | 25.08 / 24.91 | 0.8660 |
| 100 | 2.313e-03 | 27.21 / 27.08 | 0.8841 |
| 200 | 1.444e-03 | 29.27 / 29.23 | 0.8992 |

### Reconstruction Conclusion

Hardcap90 did **not** degrade reconstruction.

In fact, in the notebook comparison, hardcap90 reconstructed substantially better than the vanilla checkpoint being compared.

This is important because the improved stability is not caused by underfitting.

---

## 8. Modulation Sensitivity Result

We compared vanilla vs hardcap90 using:

```text
A(x, δ) = ||φ(x + δ) − φ(x)||₂ / ||δ||₂
```

Lower is better.

---

## 9. Sensitivity Reduction at ε = 8/255

At `eps = 8/255`, hardcap90 consistently reduced modulation sensitivity across all inner-loop budgets.

| Fit steps | Vanilla mean_A | Hardcap90 mean_A | Reduction |
|---:|---:|---:|---:|
| 5 | 0.004092 | 0.002688 | 34.3% lower |
| 20 | 0.008401 | 0.005424 | 35.4% lower |
| 50 | 0.014636 | 0.009621 | 34.3% lower |
| 100 | 0.022418 | 0.014757 | 34.2% lower |
| 200 | 0.033187 | 0.022209 | 33.1% lower |

### Main Sensitivity Conclusion at ε = 8/255

```text
Hardcap90 reduced modulation sensitivity by about one third.
```

This is a very clean and consistent result.

---

## 10. Sensitivity Reduction at ε = 16/255

At `eps = 16/255`, the reduction remained positive.

| Fit steps | Vanilla mean_A | Hardcap90 mean_A | Reduction |
|---:|---:|---:|---:|
| 5 | 0.003734 | 0.002657 | 28.8% lower |
| 20 | 0.008256 | 0.005313 | 35.6% lower |
| 50 | 0.013823 | 0.009741 | 29.5% lower |
| 100 | 0.020098 | 0.015096 | 24.9% lower |
| 200 | 0.027956 | 0.022743 | 18.6% lower |

### Main Sensitivity Conclusion at ε = 16/255

```text
Hardcap90 reduced modulation sensitivity by roughly 19% to 36%,
depending on the inner-loop fitting budget.
```

---

## 11. Interpretation of Sensitivity Results

The hardcap90 model reduced:

```text
||φ(x + δ) − φ(x)||₂
```

for the same perturbation size:

```text
||δ||₂
```

This means that the fitted modulation vectors are more stable under image perturbations.

This directly supports the desired robustness mechanism:

```text
smaller image perturbation amplification into modulation space
   ↓
less movement across classifier decision boundaries
   ↓
better adversarial robustness
```

---

## 12. PGD Robustness: Hardcap90

We then ran Full-PGD signal-space attacks on hardcap90.

### Hardcap90 PGD Summary

| eps (/255) | n | clean acc | robust acc | robust \| clean | gap |
|---:|---:|---:|---:|---:|---:|
| 8 | 200 | 0.9750 | 0.9400 | 0.9641 | +0.0350 |
| 16 | 500 | 0.9680 | 0.8460 | 0.8740 | +0.1220 |
| 32 | 200 | 0.9750 | 0.4150 | 0.4256 | +0.5600 |
| 64 | 200 | 0.9750 | 0.0300 | 0.0308 | +0.9450 |

### Hardcap90 PGD Interpretation

Hardcap90 is strong at moderate perturbation budgets:

```text
ε = 8/255  → robust accuracy = 94.0%
ε = 16/255 → robust accuracy = 84.6%
```

Robustness drops at larger budgets:

```text
ε = 32/255 → robust accuracy = 41.5%
ε = 64/255 → robust accuracy = 3.0%
```

This is expected because `32/255` and especially `64/255` are very large perturbation budgets for MNIST.

---

## 13. PGD Comparison: Vanilla vs Hardcap90

We compared hardcap90 to the vanilla model using the same Full-PGD settings.

### Vanilla PGD Summary

| eps (/255) | n | clean acc | robust acc | robust \| clean | gap |
|---:|---:|---:|---:|---:|---:|
| 8 | 200 | 0.9900 | 0.9500 | 0.9596 | +0.0400 |
| 16 | 500 | 0.9720 | 0.7120 | 0.7325 | +0.2600 |
| 32 | 200 | 0.9900 | 0.2150 | 0.2172 | +0.7750 |
| 64 | 200 | 0.9900 | 0.0300 | 0.0303 | +0.9600 |

### Hardcap90 PGD Summary

| eps (/255) | n | clean acc | robust acc | robust \| clean | gap |
|---:|---:|---:|---:|---:|---:|
| 8 | 200 | 0.9750 | 0.9400 | 0.9641 | +0.0350 |
| 16 | 500 | 0.9680 | 0.8460 | 0.8740 | +0.1220 |
| 32 | 200 | 0.9750 | 0.4150 | 0.4256 | +0.5600 |
| 64 | 200 | 0.9750 | 0.0300 | 0.0308 | +0.9450 |

---

## 14. Robust Accuracy Improvement

| eps (/255) | Vanilla robust acc | Hardcap90 robust acc | Improvement |
|---:|---:|---:|---:|
| 8 | 0.9500 | 0.9400 | -0.0100 |
| 16 | 0.7120 | 0.8460 | +0.1340 |
| 32 | 0.2150 | 0.4150 | +0.2000 |
| 64 | 0.0300 | 0.0300 | +0.0000 |

### Main PGD Result

Hardcap90 substantially improves robustness at medium perturbation budgets:

```text
ε = 16/255 → +13.4 percentage points robust accuracy
ε = 32/255 → +20.0 percentage points robust accuracy
```

At `8/255`, the robust accuracy is slightly lower, but the hardcap90 clean accuracy is also lower.

At `64/255`, both models are essentially broken.

---

## 15. Conditional Robustness Improvement

Conditional robustness only considers samples that were classified correctly before attack.

| eps (/255) | Vanilla robust\|clean | Hardcap90 robust\|clean | Improvement |
|---:|---:|---:|---:|
| 8 | 0.9596 | 0.9641 | +0.0045 |
| 16 | 0.7325 | 0.8740 | +0.1415 |
| 32 | 0.2172 | 0.4256 | +0.2084 |
| 64 | 0.0303 | 0.0308 | +0.0005 |

### Conditional Robustness Conclusion

Conditioned on clean-correct samples, hardcap90 improves robustness at every tested epsilon.

The main gains are:

```text
ε = 16/255 → +14.15 percentage points
ε = 32/255 → +20.84 percentage points
```

This is the cleanest robustness comparison because it accounts for the slightly different clean accuracies.

---

## 16. Clean Accuracy Tradeoff

Hardcap90 has slightly lower clean accuracy than vanilla.

Examples:

```text
At n = 500:
vanilla clean acc   = 97.2%
hardcap90 clean acc = 96.8%

At n = 200:
vanilla clean acc   = 99.0%
hardcap90 clean acc = 97.5%
```

So hardcap90 sacrifices approximately:

```text
0.4% to 1.5% clean accuracy
```

depending on subset.

But it gains much more robust accuracy at `16/255` and `32/255`.

This is a favorable robustness tradeoff.

---

## 17. Main Experiment Conclusions

### Conclusion 1

The previous aggressive global soft-Lipschitz setting was too strong and harmed the representation.

### Conclusion 2

A mild, per-layer cap calibrated to the vanilla spectrum works much better.

### Conclusion 3

Hardcap90 reduced modulation sensitivity by about:

```text
33–35% at ε = 8/255
19–36% at ε = 16/255
```

depending on fitting steps.

### Conclusion 4

Hardcap90 preserved strong reconstruction quality and strong downstream classification accuracy.

### Conclusion 5

Hardcap90 improved PGD robustness substantially at important medium budgets:

```text
+13.4 percentage points robust accuracy at ε = 16/255
+20.0 percentage points robust accuracy at ε = 32/255
```

### Conclusion 6

Conditioned on clean-correct samples, the robustness improvement is even clearer:

```text
+14.15 percentage points at ε = 16/255
+20.84 percentage points at ε = 32/255
```

### Conclusion 7

At very high perturbation budgets like `64/255`, both vanilla and hardcap90 fail.

---

## 18. Thesis-Level Summary

A good thesis-level interpretation is:

> The hardcap90 per-layer soft-Lipschitz model demonstrates that carefully calibrated spectral regularization can improve the robustness of parameter-space classifiers. Unlike the earlier global soft-Lipschitz configuration, which was too aggressive and increased modulation sensitivity, hardcap90 uses mild per-layer caps based on the vanilla spectrum. This reduced the sensitivity of fitted modulation vectors to image perturbations by roughly one third at ε=8/255, while preserving strong reconstruction quality and classifier accuracy. Under Full-PGD attacks, hardcap90 improved robust accuracy from 71.2% to 84.6% at ε=16/255 and from 21.5% to 41.5% at ε=32/255. These results suggest that stabilizing the image-to-modulation fitting process is a viable direction for improving adversarial robustness in parameter-space classification pipelines.

---

## 19. Important Caveats

1. The improved robustness should still be tested across more random seeds.
2. The same data subset should be used for all comparisons where possible.
3. PGD strength should be validated to reduce the risk of gradient obfuscation.
4. Additional attacks may be needed:
   - more PGD steps,
   - different PGD learning rates,
   - random restarts,
   - black-box attacks,
   - transfer attacks.
5. The modulation matrix grew during training, so future work should analyze whether mild modulation regularization can improve robustness further.

---

## 20. Recommended Next Experiments

### 20.1 Repeat With Multiple Seeds

Run hardcap90 with different seeds to verify the effect is robust.

### 20.2 Add Mild Modulation Cap

Since the modulation matrix grew during training, test:

```text
hardcap90 SIREN caps
+
weak modul cap
```

For example:

```text
modul cap ≈ 110% or 120% of vanilla
```

not an aggressive cap.

### 20.3 Try Different Cap Strengths

Run a small sweep:

```text
hardcap95
hardcap90
hardcap85
hardcap80
```

This would show the tradeoff between reconstruction, sensitivity, and robustness.

### 20.4 Validate PGD Strength

Test:

```text
PGD_STEPS = 100, 200
PGD_MOD_STEPS = 10, 20
multiple random restarts
different PGD learning rates
```

The goal is to ensure the robustness improvement is real and not caused by weak gradients.

### 20.5 Measure Classifier-Side Sensitivity

Measure:

```text
||C(φ(x + δ)) − C(φ(x))|| / ||δ||
```

and:

```text
||C(φ(x + δ)) − C(φ(x))|| / ||φ(x + δ) − φ(x)||
```

This separates:

```text
image → modulation stability
```

from:

```text
modulation → classifier stability
```

### 20.6 Analyze Jacobian Conditioning

Measure the Jacobian:

```text
J = ∂Fθ(φ) / ∂φ
```

and compare:

```text
σmax(J)
σmin(J)
condition number
```

for vanilla vs hardcap90.

This can help explain why hardcap90 improves inverse/fitting stability.

---

# Final One-Sentence Conclusion

The hardcap90 per-layer soft-Lipschitz model is the first configuration in this project that simultaneously improves modulation stability, preserves reconstruction/classification quality, and significantly improves Full-PGD robustness at medium perturbation budgets.
