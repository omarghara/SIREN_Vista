# Hard Spectral Projection for Robust SIREN/Functa Experiments

## 1. Background

In the current SIREN/Functa robustness experiments, we tried to make the model more robust by adding spectral constraints/caps to selected layers. The goal was to reduce adversarial amplification across the pipeline.

The diagnostic notebook `cifar10_latest_robustness_layer_analysis.ipynb` measures whether a small adversarial image perturbation creates large internal activation changes. For a clean image and an adversarial image,

\[
x_{\text{adv}} = x + \delta, \qquad \|\delta\|_{\infty} \leq \epsilon,
\]

the notebook fits two modulations:

\[
\phi_{\text{clean}} \approx \arg\min_{\phi} \|f_\theta(\phi) - x\|_2^2
\]

and

\[
\phi_{\text{adv}} \approx \arg\min_{\phi} \|f_\theta(\phi) - x_{\text{adv}}\|_2^2.
\]

Then it compares activations layer by layer:

\[
\Delta a_l = a_l^{\text{adv}} - a_l^{\text{clean}}.
\]

The key plotted quantity is:

\[
\|\Delta a_l\|_2 = \|a_l^{\text{adv}} - a_l^{\text{clean}}\|_2.
\]

The notebook also computes a layer amplification ratio:

\[
R_l =
\frac{\|a_l^{\text{adv}} - a_l^{\text{clean}}\|_2}
{\|a_{l-1}^{\text{adv}} - a_{l-1}^{\text{clean}}\|_2}.
\]

If \(R_l > 1\), layer \(l\) amplifies the adversarial difference. If \(R_l < 1\), layer \(l\) attenuates it.

The current result seems to show that the existing cap/penalty did not clearly reduce adversarial amplification.

---

## 2. Current cap: soft penalty

The current implementation uses a soft spectral penalty of the form:

\[
L_{\text{total}}
=
L_{\text{task}}
+
\lambda
\sum_l
\max(0, \sigma_{\max}(W_l) - c_l)^2.
\]

Here:

- \(W_l\) is the weight matrix of layer \(l\),
- \(\sigma_{\max}(W_l)\) is the largest singular value of \(W_l\),
- \(c_l\) is the desired cap,
- \(\lambda\) controls how strongly we penalize cap violations.

This method encourages the model to keep \(\sigma_{\max}(W_l)\leq c_l\), but it does not guarantee it.

The optimizer may still prefer to violate the cap if doing so improves reconstruction/classification loss enough.

So the current cap is a soft constraint:

\[
\sigma_{\max}(W_l) \leq c_l
\]

is encouraged, but not enforced.

---

## 3. Proposed change: hard SVD projection

The advisor suggested using projection. The simplest projection is based on SVD.

For a weight matrix \(W\), compute:

\[
W = U \Sigma V^\top.
\]

The diagonal matrix \(\Sigma\) contains the singular values:

\[
\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \ldots).
\]

To enforce a spectral cap \(c\), clamp the singular values:

\[
\sigma_i' = \min(\sigma_i, c).
\]

Then reconstruct:

\[
W_{\text{proj}} = U \Sigma' V^\top.
\]

After projection:

\[
\sigma_{\max}(W_{\text{proj}}) \leq c.
\]

This is different from the current penalty. Instead of only adding a loss term, we force the weight matrix back into the allowed spectral-norm ball after each optimizer update.

The training step becomes:

\[
\theta_{t+1/2}
=
\operatorname{AdamStep}(\theta_t, \nabla_\theta L)
\]

then

\[
\theta_{t+1}
=
\Pi_{\mathcal{C}}(\theta_{t+1/2}),
\]

where

\[
\mathcal{C}
=
\{W : \sigma_{\max}(W) \leq c\}.
\]

---

## 4. Why this is relevant for SIREN

A SIREN layer has the form:

\[
h_l =
\sin(\omega_0(W_l h_{l-1} + b_l + s_l)).
\]

Its Jacobian with respect to the previous activation is:

\[
\frac{\partial h_l}{\partial h_{l-1}}
=
\operatorname{diag}(\cos(\cdot)) \omega_0 W_l.
\]

Because

\[
|\cos(\cdot)| \leq 1,
\]

we get the bound:

\[
\left\|
\frac{\partial h_l}{\partial h_{l-1}}
\right\|_2
\leq
\omega_0 \sigma_{\max}(W_l).
\]

Therefore, if we want a sine layer to have effective Lipschitz bound \(L_l\), the raw weight cap should be:

\[
\sigma_{\max}(W_l) \leq \frac{L_l}{\omega_0}.
\]

For the final linear readout layer, there is no sine frequency multiplier, so the cap is directly:

\[
\sigma_{\max}(W_{\text{out}}) \leq L_{\text{out}}.
\]

---

## 5. Why the previous cap may not have helped the activation plots

The activation diagnostic measures the full path:

\[
x
\rightarrow
\phi^*(x)
\rightarrow
\operatorname{modul}(\phi)
\rightarrow
\text{SIREN shifts}
\rightarrow
\text{SIREN activations}
\rightarrow
\text{classifier}.
\]

But the current cap was mostly applied to selected late layers such as:

- readout,
- pre-readout,
- readout + pre-readout.

This may be too late.

If adversarial amplification happens earlier, a readout-only cap cannot fix it.

Also, in the spatial Functa setup, the modulation matrix `modul` maps the latent spatial code into per-layer shifts. If `modul` is not constrained, a small change in \(\phi\) can become a large change in the injected shifts.

So one possible explanation is:

\[
\sigma(W_{\text{readout}}) \downarrow
\not\Rightarrow
\|a_l^{\text{adv}} - a_l^{\text{clean}}\|_2 \downarrow
\]

because the perturbation can be amplified before the readout.

---

## 6. Implementation goal

Implement a new hard projection mechanism that can be enabled during training.

The goal is to support experiments such as:

1. Hard projection on readout only.
2. Hard projection on pre-readout only.
3. Hard projection on readout + pre-readout.
4. Hard projection on all SIREN affine layers + readout.
5. Hard projection on all SIREN affine layers + readout + `modul`.

The last experiment is important because the notebook suggests adversarial movement may enter through:

\[
\phi
\rightarrow
\operatorname{modul}(\phi)
\rightarrow
\text{SIREN shifts}.
\]

---

## 7. Suggested code structure

Add a reusable projection utility, for example:

```python
@torch.no_grad()
def project_weight_svd_(weight: torch.Tensor, cap: float) -> float:
    """Project a weight tensor onto the spectral-norm ball.

    Args:
        weight: weight tensor, usually shape [out_features, in_features].
        cap: maximum allowed spectral norm.

    Returns:
        The pre-projection top singular value.
    """
    original_shape = weight.shape
    W = weight.detach().float().reshape(original_shape[0], -1)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    sigma_before = float(S[0].item())

    S_clamped = torch.clamp(S, max=cap)
    W_proj = (U * S_clamped.unsqueeze(0)) @ Vh

    weight.copy_(W_proj.to(dtype=weight.dtype).reshape(original_shape))
    return sigma_before
```

Then add a model-level function:

```python
@torch.no_grad()
def project_model_spectral_caps_(model, config) -> dict:
    """Apply SVD projection to selected model layers.

    Returns a log dictionary with pre/post singular values.
    """
```

The config should specify:

- which layers are projected,
- the cap value for each layer,
- whether sine-layer caps should be divided by `freq`,
- whether to include `modul`.

---

## 8. Where to call projection

Projection should happen immediately after the optimizer update.

Current training is roughly:

```python
outer_optimizer.zero_grad()
outer_loss.backward()
outer_optimizer.step()
```

The projected version should be:

```python
outer_optimizer.zero_grad()
outer_loss.backward()
outer_optimizer.step()

if use_svd_projection:
    projection_stats = project_model_spectral_caps_(modSiren, projection_config)
```

This is projected gradient descent:

\[
\theta_{t+1/2}
=
\operatorname{OptimizerStep}(\theta_t)
\]

\[
\theta_{t+1}
=
\Pi_{\mathcal{C}}(\theta_{t+1/2}).
\]

---

## 9. Important implementation details

### 9.1 Do not project biases

Only project weight matrices.

Biases do not affect the spectral norm of the linear map.

### 9.2 Preserve dtype and device

The SVD may be safer in float32, but the projected weight should be copied back to the original dtype and device.

### 9.3 Log before/after singular values

For debugging, log:

\[
\sigma_{\max}(W_{\text{before}})
\]

and

\[
\sigma_{\max}(W_{\text{after}}).
\]

The after value should be at most the cap, up to small numerical tolerance.

### 9.4 Full SVD is okay for now

The model is small enough that exact SVD is acceptable for a first implementation.

Later, if training becomes too slow, we can project less frequently or use approximate methods.

### 9.5 Frequency-adjusted cap for sine layers

For sine layers:

\[
\text{raw cap} = \frac{L}{\omega_0}.
\]

For readout and `modul`, use the cap directly unless we explicitly define a different convention.

---

## 10. Suggested experiment names

Use clear run labels, for example:

- `svdproj_readout_cap10`
- `svdproj_prereadout_cap10`
- `svdproj_readout_prereadout_cap10`
- `svdproj_all_sine_readout_L1`
- `svdproj_all_sine_readout_modul_L1`

The exact naming should match the existing repo style.

---

## 11. Diagnostics after implementation

After training each projected model, run the existing notebook again:

`cifar10_latest_robustness_layer_analysis.ipynb`

Check these quantities:

### 11.1 Checkpoint singular values

Verify:

\[
\sigma_{\max}(W_l) \leq c_l.
\]

If this is false, projection is not working.

### 11.2 Product bound

Check whether:

\[
\prod_l \omega_0 \sigma(W_l) \cdot \sigma(W_{\text{out}})
\]

is actually reduced.

### 11.3 Activation difference

Check whether:

\[
\|a_l^{\text{adv}} - a_l^{\text{clean}}\|_2
\]

is lower than vanilla.

### 11.4 Normalized activation difference

Check:

\[
\frac{\|a_l^{\text{adv}} - a_l^{\text{clean}}\|_2}{\|\delta\|_2}.
\]

This is an empirical gain from input perturbation to layer activation.

### 11.5 Layer amplification ratio

Check:

\[
R_l =
\frac{\|\Delta a_l\|_2}{\|\Delta a_{l-1}\|_2}.
\]

If \(R_l > 1\) still occurs strongly in early layers, then the projected layers may still be too late or incomplete.

---

## 12. Acceptance criteria

The implementation is considered correct if:

1. A training flag/config can enable hard SVD projection.
2. Projection can target at least:
   - readout,
   - pre-readout,
   - all SIREN affine layers,
   - optionally `modul`.
3. Projection is applied after `outer_optimizer.step()`.
4. A post-training checkpoint sigma script/notebook shows projected layers satisfy:

\[
\sigma_{\max}(W_l) \leq c_l + 10^{-5}.
\]

5. Existing training still works when projection is disabled.
6. The robustness layer-analysis notebook can compare projected runs against the existing vanilla/soft-cap runs.

---

## 13. Main hypothesis

The hypothesis is:

> A hard SVD projection will enforce the spectral cap more reliably than the current soft penalty, and broader projection over earlier SIREN layers and/or the modulation matrix may reduce adversarial activation amplification more than readout-only caps.

This hypothesis may still be false experimentally, but the proposed implementation will give a cleaner test than the current soft penalty.
