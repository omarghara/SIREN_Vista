> **Note (outdated in part).** This file snapshots two earlier walk-throughs of
> the variants package and `soft_lipschitz.py`. The control-flow and math
> explanations remain accurate, but the specific soft-Lipschitz **cap
> semantics** described here are pre-update: `--soft-lip-cap` used to be a
> raw per-layer σ-cap `c`, with slug prefix `softlip_c*`. The current meaning
> is a **per-layer Lipschitz budget `L`** (sine layers get σ-cap `L/freq`,
> readout gets `L`), with slug prefix `softlip_L*`. See `CHANGES.md` §7 for
> the full update and `SIREN_Vista/variants/soft_lipschitz.py` for the
> live code.

---

## 1. The mental model

Think of `variants/` as a **mini plug-in system**. Every research idea becomes one file that implements a 4-method contract:

| method | when the trainer calls it |
|---|---|
| `add_args(parser)` | at CLI parse time, to declare its own flags |
| `build(base_model, args)` | right after the base `ModulatedSIREN` is constructed, to optionally modify the architecture |
| `penalty(model, args)` | every batch, returns a scalar tensor added to the outer loss |
| `slug(args)` | when picking the save directory name |

The trainer itself doesn't know anything about soft-Lipschitz specifically. It only knows "call these 4 methods on whatever variant the user picked."

---

## 2. Boot phase — how the registry gets populated

Start with `variants/__init__.py`:

```9:18:SIREN_Vista/variants/__init__.py
REGISTRY = {}


def register(name):
    def deco(cls):
        assert name not in REGISTRY, f"variant '{name}' already registered"
        REGISTRY[name] = cls
        cls.variant_name = name
        return cls
    return deco
```

`REGISTRY` is a module-level dict. `@register("foo")` is a class decorator — when Python sees the decorator, it runs `deco(cls)`, which stores the class in `REGISTRY["foo"]` and returns it unchanged.

The magic line is at the bottom:

```61:61:SIREN_Vista/variants/__init__.py
from . import vanilla, soft_lipschitz  # noqa: F401, E402 — registers on import
```

When anyone does `import variants`, Python executes `__init__.py` top-to-bottom. That last line imports `variants/vanilla.py` and `variants/soft_lipschitz.py`. Each of those files has a `@register(...)` decorator at module scope, and the decorator runs *during import*. So by the time `import variants` returns, `REGISTRY` already contains both names.

You can verify this mentally: look at `variants/vanilla.py`:

```8:9:SIREN_Vista/variants/vanilla.py
@register("vanilla")
class Vanilla:
```

That decorator line fires during import, putting `Vanilla` into `REGISTRY["vanilla"]`.

Same for soft-Lipschitz:

```22:23:SIREN_Vista/variants/soft_lipschitz.py
@register("soft_lipschitz")
class SoftLipschitz:
```

So `REGISTRY` ends up as:

```python
{"vanilla": Vanilla, "soft_lipschitz": SoftLipschitz}
```

The dispatch functions are just one-liners that look up the class by name and delegate to its static method:

```34:43:SIREN_Vista/variants/__init__.py
def build(name, base_model, args):
    return REGISTRY[name].build(base_model, args)


def penalty(name, model, args):
    return REGISTRY[name].penalty(model, args)


def slug(name, args):
    return REGISTRY[name].slug(args)
```

That's the whole registry. No metaclasses, no deep magic — just a dict of classes.

---

## 3. CLI construction — how the flags appear

Now follow `trainer.py` from the top. First there's the new import:

```12:12:SIREN_Vista/trainer.py
import variants
```

Just importing this triggers the chain in section 2 — by the time the next line runs, both variants are registered.

Inside `get_args()`:

```110:115:SIREN_Vista/trainer.py
    parser.add_argument('--variant', choices=variants.available(), default='vanilla',
                        help='SIREN variant to train.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional subdirectory name override under '
                             'model_{dataset}/. Defaults to the variant slug.')
    variants.add_all_variant_args(parser)
```

- `variants.available()` returns `["soft_lipschitz", "vanilla"]`, and argparse uses that list as the allowed `choices`.
- `variants.add_all_variant_args(parser)` loops over every registered class and calls its `add_args`:

```29:31:SIREN_Vista/variants/__init__.py
def add_all_variant_args(parser):
    for cls in REGISTRY.values():
        cls.add_args(parser)
```

For `Vanilla`, `add_args` is a `pass` — contributes nothing. For `SoftLipschitz`, `add_args` creates an argparse group and appends the four `--soft-lip-*` flags:

```24:42:SIREN_Vista/variants/soft_lipschitz.py
    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group("soft_lipschitz")
        g.add_argument("--soft-lip-cap", type=float, default=1.0,
                       help="per-layer spectral-norm target c. "
                            "Penalty is max(0, ||W||_2 - c)^2.")
        g.add_argument("--soft-lip-lambda", type=float, default=1e-2,
                       help="penalty weight lambda.")
        g.add_argument("--soft-lip-apply-to",
                       choices=["sine_only", "sine_and_readout", "all"],
                       default="sine_only",
                       help="which SIREN linear layers to penalize. "
                            "'sine_only' = the W inside sin(omega_0 (W x + b + s)); "
                            "'sine_and_readout' additionally penalizes the final "
                            "hidden2rgb linear; 'all' also penalizes the modulation "
                            "linear mapping phi -> per-layer shifts.")
        g.add_argument("--soft-lip-power-iters", type=int, default=1,
                       help="power-iteration steps per batch for the "
                            "spectral-norm estimate.")
```

Important detail: argparse converts dashes to underscores in attribute names. So after parsing, the `args` namespace carries `args.soft_lip_cap`, `args.soft_lip_lambda`, `args.soft_lip_apply_to`, `args.soft_lip_power_iters`. This is the naming convention the penalty code relies on later.

After `args = get_args()` returns, every variant's flags live on `args` — even the ones for variants you're *not* using. That's harmless; `Vanilla.penalty` never reads any of them.

---

## 4. Main flow — trainer.py `__main__`, line by line

Let me walk through the `__main__` block:

```121:121:SIREN_Vista/trainer.py
    args = get_args()
```

Now `args.variant` is either `'vanilla'` or `'soft_lipschitz'`, plus all the variant-specific fields.

```125:133:SIREN_Vista/trainer.py
    if args.dataset == "modelnet":
        resample_shape = (15,15,15) #we use this resampling in all experiments
        dataloader = get_modelnet_loader(train=True, batch_size=args.batch_size, resample_shape=resample_shape)
        modSiren = ModulatedSIREN3D(height=resample_shape[0], width=resample_shape[1], depth=resample_shape[2],\
            hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #we use a mod dim of 2048 in our exps
  
    else:        
        dataloader = get_mnist_loader(args.data_path, train=True, batch_size=args.batch_size, fashion = args.dataset=="fmnist")
        modSiren = ModulatedSIREN(height=28, width=28, hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #28,28 is mnist and fmnist dims
```

This is unchanged from before — builds the *base* `ModulatedSIREN` or `ModulatedSIREN3D`.

```138:142:SIREN_Vista/trainer.py
    modSiren = modSiren.to(args.device)
    modSiren = variants.build(args.variant, modSiren, args)
    optimizer = optim.Adam(modSiren.parameters(), lr=args.ext_lr)
    criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    penalty_fn = lambda m: variants.penalty(args.variant, m, args)
```

Three things happen here, in order:

1. **Line 139 — `variants.build(...)`** — dispatches to `Vanilla.build` or `SoftLipschitz.build`. Both just `return base_model` right now (nothing to wrap). But this is the hook point where a future hard-Lipschitz variant would do `torch.nn.utils.spectral_norm(layer)` on each `SineAffine.affine`, which would add `weight_orig`/`weight_u`/`weight_v` parameters. The optimizer on line 140 is constructed *after* `build`, so any parameters added by a variant get registered with the optimizer automatically.

2. **Line 140 — Optimizer** — created after build so it sees all parameters, including any added by the variant.

3. **Line 142 — `penalty_fn`** — a closure. It freezes `args.variant` and `args` now, so later calls `penalty_fn(model)` need only the model to work. This closure is passed into `fit()` as a kwarg.

Then the resume block:

```144:162:SIREN_Vista/trainer.py
    start_epoch = 0
    best_loss = float('Inf')
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        modSiren.load_state_dict(ckpt['state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print(f"[resume] '{args.resume}' has no optimizer_state_dict; "
                  f"starting optimizer fresh.")
        start_epoch = ckpt.get('epoch', -1) + 1
        best_loss   = ckpt.get('loss', float('Inf'))
        ckpt_variant = ckpt.get('variant')
        if ckpt_variant is not None and ckpt_variant != args.variant:
            print(f"[resume] WARNING: checkpoint variant '{ckpt_variant}' "
                  f"differs from requested --variant '{args.variant}'. "
                  f"Continuing with '{args.variant}'.")
        print(f"[resume] loaded '{args.resume}' at epoch {start_epoch-1}, "
              f"best_loss={best_loss:.6f}")
```

Nothing variant-specific here except the mismatch warning at lines 156–160. Resume loads state into whatever architecture `variants.build` just produced, which is why `build` must be called *before* `load_state_dict`.

Then the save-path decision:

```164:168:SIREN_Vista/trainer.py
    if args.run_name is not None:
        run_slug = args.run_name
    else:
        run_slug = variants.slug(args.variant, args)
    savedir = f"model_{args.dataset}/{run_slug}" if run_slug else f"model_{args.dataset}"
```

`variants.slug(args.variant, args)` dispatches to `Vanilla.slug` (returns `"vanilla"`) or `SoftLipschitz.slug` (builds something like `"softlip_c1_lam1e-02_sine_only"`). Result: `savedir = "model_mnist/softlip_c1_lam1e-02_sine_only"`.

Then the epoch loop:

```171:175:SIREN_Vista/trainer.py
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        loss = fit(
            modSiren, dataloader, optimizer, criterion, epoch, inner_steps=3,inner_lr=args.int_lr, voxels=args.dataset=='modelnet',
            penalty_fn=penalty_fn,
        )
```

The `penalty_fn=penalty_fn` kwarg is the hook.

Then checkpoint save with metadata:

```176:190:SIREN_Vista/trainer.py
        if loss < best_loss:
            best_loss = loss
            torch.save({'epoch': epoch,
                        'state_dict': modSiren.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'variant': args.variant,
                        'variant_args': variants._extract_variant_args(args, args.variant),
                        'model_args': {
                            'dataset': args.dataset,
                            'hidden_dim': args.hidden_dim,
                            'mod_dim': args.mod_dim,
                            'depth': args.depth,
                        },
                        }, f'{savedir}/modSiren.pth')
```

`variants._extract_variant_args(args, args.variant)` returns a compact dict of *only* the flags belonging to that variant:

```46:58:SIREN_Vista/variants/__init__.py
def _extract_variant_args(args, name):
    """Return a dict with only the CLI attributes whose names start with the
    variant's argparse prefix. The prefix is inferred from the variant name
    by lowercasing and replacing underscores with the short form used in
    argparse dest (e.g. ``soft_lipschitz`` -> ``soft_lip_``)."""
    prefix_map = {
        "vanilla": None,
        "soft_lipschitz": "soft_lip_",
    }
    prefix = prefix_map.get(name)
    if prefix is None:
        return {}
    return {k: v for k, v in vars(args).items() if k.startswith(prefix)}
```

So for soft-Lipschitz you get `{"soft_lip_cap": 1.0, "soft_lip_lambda": 0.01, "soft_lip_apply_to": "sine_only", "soft_lip_power_iters": 1}` saved in the `.pth`. For vanilla, an empty dict.

---

## 5. Inside `fit()` — where the penalty is applied

The only variant-aware lines inside `fit()` are:

```70:80:SIREN_Vista/trainer.py
        outer_optimizer.zero_grad()
        outer_loss = torch.tensor(0).to(device).float()
        for batch_id in range(batch_size):
            modulator = modulators[batch_id]
            # Outer Optimization.
            fitted = model(modulator)
            outer_loss += (outer_criterion(fitted.T, images[batch_id].flatten()[None]) if voxels else outer_criterion(fitted, images[batch_id])) / batch_size
        if penalty_fn is not None:
            outer_loss = outer_loss + penalty_fn(model)
        # Outer optimizer step.
        outer_loss.backward()
```

Flow per batch:
1. Outer loss starts at 0.
2. For each image in the batch, reconstruct with the current modulator and add MSE to outer loss.
3. **Line 77–78** — `penalty_fn(model)` is called. For vanilla that returns `torch.zeros(())` so the outer loss is unchanged. For soft-Lipschitz it returns `λ * Σ_l max(0, σ_l - c)^2`, which is a scalar tensor that *does* require grad, meaning `outer_loss.backward()` will propagate gradients through it into each penalized `W_l`.
4. `outer_loss.backward()` — one backward pass for reconstruction + penalty combined.
5. Gradient clipping, optimizer step.

The penalty is added *once per batch* (outside the inner for-loop), not per sample. Same training cost structure as before plus a small constant from power iteration.

---

## 6. How the penalty actually computes — the clever part

This is the most subtle piece. Read `variants/soft_lipschitz.py` alongside.

### 6a. Layer selection

When `penalty()` is called:

```48:58:SIREN_Vista/variants/soft_lipschitz.py
    @staticmethod
    def penalty(model, args):
        layers = _collect_layers(model, args.soft_lip_apply_to)
        if not layers:
            return torch.zeros((), device=next(model.parameters()).device)
        sigmas = [_power_iter_sigma(lin, n_iter=args.soft_lip_power_iters)
                  for lin in layers]
        excess = torch.stack([
            torch.clamp(s - args.soft_lip_cap, min=0.0) ** 2 for s in sigmas
        ])
        return args.soft_lip_lambda * excess.sum()
```

`_collect_layers` walks `model.modules()` and picks out the right `nn.Linear` instances:

```67:88:SIREN_Vista/variants/soft_lipschitz.py
def _collect_layers(model, mode):
    """Return the list of nn.Linear modules to penalize.

    ``sine_only``         -> SineAffine.affine layers only.
    ``sine_and_readout``  -> + SIREN.hidden2rgb.
    ``all``               -> + ModulatedSIREN(.3D).modul.
    """
    lins = []
    for m in model.modules():
        if isinstance(m, SineAffine):
            lins.append(m.affine)

    if mode in ("sine_and_readout", "all"):
        siren = getattr(model, "siren", None)
        if siren is not None and hasattr(siren, "hidden2rgb"):
            lins.append(siren.hidden2rgb)

    if mode == "all":
        if isinstance(model, (ModulatedSIREN, ModulatedSIREN3D)):
            lins.append(model.modul)

    return lins
```

For a `ModulatedSIREN(depth=4)`:
- `sine_only` → 4 layers (each `SineAffine`'s internal `affine`)
- `sine_and_readout` → 4 + 1 = 5 (add `hidden2rgb`)
- `all` → 5 + 1 = 6 (add `modul`)

### 6b. Differentiable spectral norm via power iteration

This is the heart of the variant. Look at these two functions carefully:

```91:117:SIREN_Vista/variants/soft_lipschitz.py
@torch.no_grad()
def _update_uv(lin, n_iter):
    """One or more power-iteration steps updating the persistent u, v buffers
    attached to ``lin``. Buffers are lazily created on first call.
    """
    W = lin.weight
    W2d = W.reshape(W.shape[0], -1)
    out_dim, in_dim = W2d.shape

    if not hasattr(lin, "_sl_u") or lin._sl_u.shape[0] != out_dim \
            or lin._sl_u.device != W.device or lin._sl_u.dtype != W.dtype:
        u = torch.randn(out_dim, device=W.device, dtype=W.dtype)
        u = u / (u.norm() + 1e-12)
        v = torch.randn(in_dim, device=W.device, dtype=W.dtype)
        v = v / (v.norm() + 1e-12)
        lin._sl_u = u
        lin._sl_v = v

    u = lin._sl_u
    v = lin._sl_v
    for _ in range(max(1, n_iter)):
        v = W2d.t() @ u
        v = v / (v.norm() + 1e-12)
        u = W2d @ v
        u = u / (u.norm() + 1e-12)
    lin._sl_u = u
    lin._sl_v = v
```

Two key ideas:

1. **Persistent buffers**. The first time `_update_uv` sees a layer it creates random unit vectors `u` (length = out dim) and `v` (length = in dim) and stashes them on the layer object itself as `_sl_u`, `_sl_v`. On subsequent calls it reuses them. This is exactly the trick `torch.nn.utils.spectral_norm` uses: one power-iter step per batch is cheap, and because `u` and `v` are preserved across batches, they quickly converge to the top left/right singular vectors of `W`.

2. **`@torch.no_grad()`**. The whole function is decorated, so PyTorch doesn't build a computation graph through the iteration. `u` and `v` are treated as plain data, not differentiable tensors.

Then:

```120:131:SIREN_Vista/variants/soft_lipschitz.py
def _power_iter_sigma(lin, n_iter=1):
    """Estimate the top singular value of ``lin.weight`` differentiably.

    Power iteration updates u, v under no-grad. Then sigma = u^T W v is
    computed with gradients w.r.t. W enabled.
    """
    _update_uv(lin, n_iter)
    W = lin.weight
    W2d = W.reshape(W.shape[0], -1)
    u = lin._sl_u
    v = lin._sl_v
    return torch.dot(u, W2d @ v)
```

Note: `_power_iter_sigma` is **not** decorated with `@torch.no_grad()`. So after `_update_uv` has produced fresh `u`, `v` (no grad), the return statement `u^T (W @ v)` treats `W` as a leaf tensor with grad, and `u`, `v` as constants. The result is:

$$\sigma = u^\top W v$$

If `u` and `v` are close to the true top singular vectors, this equals `σ_max(W)`. And crucially:

$$\frac{\partial \sigma}{\partial W} = u v^\top$$

which is a well-defined matrix, and autograd computes it automatically. **So the gradient of the penalty flows into every `W_l` that exceeded the cap.**

That's why we verified in the smoke test that after `.backward()`, all `SineAffine.affine.weight` grads were non-zero.

Back in `penalty()`:

```55:58:SIREN_Vista/variants/soft_lipschitz.py
        excess = torch.stack([
            torch.clamp(s - args.soft_lip_cap, min=0.0) ** 2 for s in sigmas
        ])
        return args.soft_lip_lambda * excess.sum()
```

Hinge-squared form: `max(0, σ − c)^2`. Layers *below* the cap contribute 0 (and the clamp's gradient is 0 there, so no gradient on those `W_l`). Layers *above* the cap contribute `2(σ − c) · dσ/dW`, pushing `σ` back down.

---

## 7. What changes for `makeset.py`

`makeset.py` is simpler — it reconstructs the model, loads the `.pth`, then fits per-sample modulations. The two variant-aware hooks are symmetric with trainer:

- `--variant` flag + `variants.add_all_variant_args(parser)` so it accepts the same CLI shape.
- `modSiren = variants.build(args.variant, modSiren, args)` is called *before* `load_state_dict`, so if a variant added parameters (future hard-Lipschitz), they exist before weights are loaded.

No penalty and no slug — this script isn't training.

---

## 8. End-to-end picture

```mermaid
sequenceDiagram
    participant CLI
    participant trainer as trainer.py
    participant reg as variants/__init__.py
    participant softlip as variants/soft_lipschitz.py
    participant model as ModulatedSIREN
    participant fit as fit loop

    CLI->>trainer: python trainer.py --variant soft_lipschitz --soft-lip-cap 1.0 ...
    trainer->>reg: import variants
    reg->>softlip: import registers @register("soft_lipschitz")
    reg-->>trainer: REGISTRY ready
    trainer->>reg: variants.add_all_variant_args(parser)
    reg->>softlip: SoftLipschitz.add_args(parser)
    softlip-->>reg: adds --soft-lip-* flags
    trainer->>trainer: args = parser.parse_args()
    trainer->>model: ModulatedSIREN(...)
    trainer->>reg: variants.build("soft_lipschitz", model, args)
    reg->>softlip: SoftLipschitz.build -> identity
    trainer->>trainer: penalty_fn = lambda m: variants.penalty(...)
    loop every batch
      trainer->>fit: outer_loss from reconstruction
      fit->>reg: penalty_fn(model)
      reg->>softlip: SoftLipschitz.penalty(model, args)
      softlip->>softlip: _collect_layers, _power_iter_sigma per layer
      softlip-->>fit: lambda * sum_l max(0, sigma_l - c)^2
      fit->>fit: outer_loss = outer_loss + penalty
      fit->>fit: outer_loss.backward(); optimizer.step()
    end
    trainer->>reg: variants.slug("soft_lipschitz", args) -> "softlip_c1_lam1e-02_sine_only"
    trainer->>trainer: torch.save(..., variant=..., variant_args=...)
```

---

## 9. Adding your next variant (how hard-Lipschitz will look)

Concretely, the next variant (hard-Lipschitz via `torch.nn.utils.spectral_norm`) would look like:

1. New file `variants/hard_lipschitz.py`:

```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from SIREN import SineAffine
from . import register

@register("hard_lipschitz")
class HardLipschitz:
    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group("hard_lipschitz")
        g.add_argument("--hard-lip-apply-to", choices=[...], default="sine_only")

    @staticmethod
    def build(base_model, args):
        for m in base_model.modules():
            if isinstance(m, SineAffine):
                spectral_norm(m.affine)         # modifies state_dict!
        return base_model

    @staticmethod
    def penalty(model, args):
        return torch.zeros((), device=next(model.parameters()).device)

    @staticmethod
    def slug(args):
        return f"hardlip_{args.hard_lip_apply_to}"
```

2. Add `hard_lipschitz` to the last-line import of `variants/__init__.py`.
3. Add its prefix to the `prefix_map` in `_extract_variant_args`.

That's the entire checklist. `trainer.py` and `makeset.py` don't change because they already call the 4-method contract generically. The `build` step this time *actually modifies* the state dict (adds `weight_orig`, `weight_u`, `weight_v`), which is why `makeset.py` must also pass `--variant hard_lipschitz` and call `variants.build(...)` before `load_state_dict` — so the keys match.

---

That's the full tour. If you want, I can next explain any sub-piece in more depth — e.g. why power iteration converges quickly, why `σ = uᵀWv` is sharp at the optimum, or what the optimizer-state implications are if a future variant adds parameters mid-training.





Alright, let me walk you through `soft_lipschitz.py` in two passes: **first the math** (why we're doing this), **then the code** (line by line).

---

# Part 1 — The math

## 1.1 Why Lipschitz at all?

A function \( f \colon \mathbb{R}^n \to \mathbb{R}^m \) is **L-Lipschitz** if, for all inputs \( x, y \),

\[
\|f(x) - f(y)\| \;\le\; L \cdot \|x - y\|.
\]

`L` is a hard upper bound on how fast the output can change. For the parameter-space classifier pipeline this matters in two places:

- **The SIREN itself, \( f_\theta(x) \)** — if Lipschitz, a small perturbation to the coordinate or modulation \( \phi \) can't cause an explosion in the reconstructed signal. That prevents pathological gradients inside the inner loop that give the illusion of adversarial robustness (the "gradient obfuscation" failure mode in the context brief).
- **The downstream classifier that reads the modulations** — robustness margins scale like \( \text{margin} / L \), so a lower Lipschitz constant gives a *certified* robustness radius in principle.

For SIREN specifically we only care about the network part, not the meshgrid input, so what we actually want to control is the Lipschitz constant of the map

\[
\phi \;\mapsto\; \text{shifts} \;\mapsto\; \text{SIREN output}.
\]

## 1.2 Lipschitz constant of the building blocks

**Linear layer.** For \( f(x) = Wx + b \),

\[
\|Wx - Wy\| \le \|W\|_2 \cdot \|x - y\|,
\]

where \( \|W\|_2 = \sigma_{\max}(W) \) is the **spectral norm** — the largest singular value of \( W \). So the tight Lipschitz constant of a linear layer is exactly \( \sigma_{\max}(W) \). The bias \( b \) adds a constant and doesn't affect Lipschitzness.

**Sine activation (the SIREN twist).** Ordinary ReLU/Tanh are 1-Lipschitz. But SIREN uses \( \sin(\omega_0 \cdot z) \) with \( \omega_0 = 30 \). Its derivative is \( \omega_0 \cos(\omega_0 z) \), bounded in magnitude by \( \omega_0 \). So

\[
\text{Lip}(\sin(\omega_0 \cdot)) \;=\; \omega_0 \;=\; 30.
\]

That 30× multiplier is what makes SIREN expressive but hard to Lipschitz-control.

**Composition.** Lipschitz constants multiply under composition:

\[
\text{Lip}(f \circ g) \;\le\; \text{Lip}(f) \cdot \text{Lip}(g).
\]

For a `SineAffine` layer \( x \mapsto \sin(\omega_0 (Wx + b + s)) \):

\[
\text{Lip}(\text{SineAffine}) \;\le\; \omega_0 \cdot \sigma_{\max}(W).
\]

For the whole modulated SIREN with layers \( 1 \ldots L \) plus the readout \( \text{hidden2rgb} \):

\[
\text{Lip}(f_\theta) \;\le\; \sigma_{\max}(W_\text{rgb}) \cdot \prod_{\ell=1}^{L} \omega_0 \cdot \sigma_{\max}(W_\ell).
\]

Notice this bound is **multiplicative** over depth. With \( \omega_0 = 30 \) and \( L = 10 \) that's a factor of \( 30^{10} \approx 6 \times 10^{14} \) before we even look at the weights. To get any meaningful bound we need each \( \sigma_{\max}(W_\ell) \) controlled. Hence the "soft cap" idea: pick a target \( c \) and keep every \( \sigma_{\max}(W_\ell) \le c \).

## 1.3 Hard vs soft constraint

Two standard ways to enforce \( \sigma_{\max}(W) \le c \):

1. **Hard** — after every step, rescale: \( W \leftarrow W \cdot \min(1,\, c / \sigma_{\max}(W)) \). This is what `torch.nn.utils.spectral_norm` does (with \( c = 1 \)). It strictly enforces the bound but clips expressivity and changes the state_dict shape.

2. **Soft** — add a differentiable penalty to the loss that punishes exceeding \( c \), and let the optimizer handle it. Keeps the architecture identical.

We implement the soft version. The natural choice is a **hinge-squared** penalty over layers \( \mathcal{L} \):

\[
\boxed{ \;\Omega(\theta) \;=\; \lambda \sum_{\ell \in \mathcal{L}} \bigl( \max(0,\; \sigma_{\max}(W_\ell) - c) \bigr)^2 \; }
\]

Three properties of this functional form:

- **Zero on the feasible side.** If \( \sigma_\ell \le c \) the term is 0 and contributes no gradient. Layers that are "well-behaved" are left alone.
- **Smooth above the threshold.** Compared to a plain hinge \( \max(0, \sigma - c) \), squaring makes the penalty \( C^1 \) at \( \sigma = c \), avoiding kinks that confuse Adam.
- **Gradient grows with violation.** \( \partial \Omega / \partial \sigma = 2\lambda(\sigma - c) \) when \( \sigma > c \), so bigger violations get pushed back harder. This is exactly the shape of a one-sided quadratic barrier.

In the code: `args.soft_lip_cap` is \( c \), `args.soft_lip_lambda` is \( \lambda \). Defaults \( c = 1.0, \lambda = 10^{-2} \).

## 1.4 Computing the spectral norm — the hard part

A naive approach: do a full SVD of every \( W_\ell \) every batch. For a 256×256 matrix that's \( O(256^3) \approx 1.6 \times 10^7 \) flops per layer per batch. Doable on MNIST, painful for anything bigger.

**Power iteration** replaces that with a cheap iterative estimator. Given a matrix \( A \in \mathbb{R}^{m \times n} \):

- Start with random unit vectors \( u_0 \in \mathbb{R}^m \), \( v_0 \in \mathbb{R}^n \).
- Iterate:
  \[
  v_{k+1} = \frac{A^\top u_k}{\|A^\top u_k\|}, \qquad u_{k+1} = \frac{A v_{k+1}}{\|A v_{k+1}\|}.
  \]
- Estimate: \( \sigma_{\max}(A) \approx u_k^\top A v_k \).

**Why it works.** Write \( A = \sum_i \sigma_i\, u_i v_i^\top \) (SVD). Then \( A^\top A = \sum_i \sigma_i^2\, v_i v_i^\top \). Applying \( A^\top A \) repeatedly to a random vector amplifies the component along the top right singular vector \( v_1 \) by \( \sigma_1^2 \) per step, and the \( i \)-th component by \( \sigma_i^2 \). After \( k \) steps, the ratio of the \( i \)-th component to the first is \( (\sigma_i / \sigma_1)^{2k} \), which decays geometrically. Convergence is exponential in the **spectral gap** \( \sigma_1 / \sigma_2 \).

At the fixed point, \( u_k = u_1 \) and \( v_k = v_1 \), and

\[
u_1^\top A v_1 \;=\; u_1^\top (\sigma_1 u_1) \;=\; \sigma_1,
\]

exactly what we want.

**The key practical trick: amortize across batches.** A single power-iter step is usually not enough to converge from scratch. But across training batches, \( W \) changes very slowly (small gradient steps), so \( v_1(W) \) also changes slowly. If we *save* \( u, v \) and warm-start next batch's iteration from them, we only need **1 iteration per batch** to stay locked on \( \sigma_1 \). This is exactly what `torch.nn.utils.spectral_norm` does and exactly what `_update_uv` does via the `_sl_u`, `_sl_v` buffers attached to each `nn.Linear`.

## 1.5 Making \( \sigma \) differentiable w.r.t. \( W \)

We need \( \partial \Omega / \partial W \) for `.backward()`. Here's the clever bit:

\[
\sigma_{\max}(W) \;=\; \max_{\|u\|=\|v\|=1}\, u^\top W v.
\]

Let \( u^*, v^* \) be the maximizers (the top singular vectors). **Danskin's envelope theorem** (or just direct calculation) says that since the constraint set doesn't depend on \( W \),

\[
\frac{\partial \sigma_{\max}(W)}{\partial W} \;=\; u^* (v^*)^\top.
\]

So if we had \( u^*, v^* \) we could write \( \sigma = (u^*)^\top W v^* \) and let autograd do the rest — it would produce exactly \( u^* (v^*)^\top \).

The soft trick in the code is: compute \( u, v \) approximately by power iteration **under `torch.no_grad()`** (so autograd treats them as constants), then compute the scalar \( \sigma = u^\top W v \) **with grad enabled**. Autograd sees:

\[
\sigma = \sum_{ij} u_i\, W_{ij}\, v_j \quad\Longrightarrow\quad \frac{\partial \sigma}{\partial W_{ij}} = u_i v_j.
\]

This is the "right" gradient *if* \( u, v \) are the true top singular vectors — and they are, approximately, because power iteration converged.

Why does this work even though we ignored \( \partial u / \partial W \) and \( \partial v / \partial W \)? Because at the maximizer, by the envelope theorem those derivative terms vanish. Outside the maximum they wouldn't, but in practice power iteration keeps us close enough that this bias is negligible. This is the standard pattern in the spectral-norm literature.

## 1.6 Putting it all together

Per batch:

1. For each penalized layer \( \ell \):
   - One power-iter step under no-grad to refresh \( u_\ell, v_\ell \).
   - Compute \( \sigma_\ell = u_\ell^\top W_\ell v_\ell \) with autograd on.
2. \( \Omega = \lambda \sum_\ell \max(0, \sigma_\ell - c)^2 \).
3. `outer_loss = MSE_reconstruction + Ω`.
4. `outer_loss.backward()` — gradients flow into \( W_\ell \) (through both paths), optimizer steps.

Cost: \( O(\sum_\ell |W_\ell|) \) per batch. On MNIST with depth 10, hidden 256 that's ~10 × 256² matvecs, negligible compared to the inner loop.

---

# Part 2 — The code

Now look at the file with the math in mind.

## 2.1 The class — public API

```22:24:SIREN_Vista/variants/soft_lipschitz.py
@register("soft_lipschitz")
class SoftLipschitz:
    @staticmethod
```

Registers the class in `variants.REGISTRY` under the key `"soft_lipschitz"`. Trainer dispatches here via `variants.penalty("soft_lipschitz", ...)`.

### `add_args` — declare hyperparameters

```24:42:SIREN_Vista/variants/soft_lipschitz.py
    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group("soft_lipschitz")
        g.add_argument("--soft-lip-cap", type=float, default=1.0,
                       help="per-layer spectral-norm target c. "
                            "Penalty is max(0, ||W||_2 - c)^2.")
        g.add_argument("--soft-lip-lambda", type=float, default=1e-2,
                       help="penalty weight lambda.")
        g.add_argument("--soft-lip-apply-to",
                       choices=["sine_only", "sine_and_readout", "all"],
                       default="sine_only",
                       help="which SIREN linear layers to penalize. "
                            "'sine_only' = the W inside sin(omega_0 (W x + b + s)); "
                            "'sine_and_readout' additionally penalizes the final "
                            "hidden2rgb linear; 'all' also penalizes the modulation "
                            "linear mapping phi -> per-layer shifts.")
        g.add_argument("--soft-lip-power-iters", type=int, default=1,
                       help="power-iteration steps per batch for the "
                            "spectral-norm estimate.")
```

Four knobs mapping directly to the math:

| flag | math symbol | default | why that default |
|---|---|---|---|
| `--soft-lip-cap` | \( c \) | 1.0 | Sensible starting point; Lipschitz grows by \( \omega_0 \) per layer even at \( c=1 \). |
| `--soft-lip-lambda` | \( \lambda \) | \( 10^{-2} \) | Small enough not to dominate reconstruction MSE at init, large enough to matter once \( \sigma_\ell \) starts exceeding \( c \). |
| `--soft-lip-apply-to` | \( \mathcal{L} \) | `sine_only` | The sine layers carry the \( \omega_0 \) blowup, so they dominate the product bound. |
| `--soft-lip-power-iters` | \( k \) | 1 | Amortized across batches — same as `spectral_norm`. |

The `add_argument_group("soft_lipschitz")` is purely cosmetic: `--help` shows flags under a labeled section.

### `build` — identity here

```44:46:SIREN_Vista/variants/soft_lipschitz.py
    @staticmethod
    def build(base_model, args):
        return base_model
```

Soft means "don't touch the architecture." Contrast with a future `hard_lipschitz` variant that would wrap each `nn.Linear` with `torch.nn.utils.spectral_norm`, adding `weight_orig`/`weight_u`/`weight_v` to the state dict.

### `penalty` — the per-batch scalar

```48:58:SIREN_Vista/variants/soft_lipschitz.py
    @staticmethod
    def penalty(model, args):
        layers = _collect_layers(model, args.soft_lip_apply_to)
        if not layers:
            return torch.zeros((), device=next(model.parameters()).device)
        sigmas = [_power_iter_sigma(lin, n_iter=args.soft_lip_power_iters)
                  for lin in layers]
        excess = torch.stack([
            torch.clamp(s - args.soft_lip_cap, min=0.0) ** 2 for s in sigmas
        ])
        return args.soft_lip_lambda * excess.sum()
```

Line-by-line translation of the formula:

1. `layers = _collect_layers(...)` — enumerate \( \mathcal{L} \).
2. Early return if empty (robustness guard; also needed if someone runs `sine_only` on a model with no `SineAffine`).
3. `sigmas[i] = _power_iter_sigma(W_i)` — this is \( \sigma_\ell \), **differentiable** w.r.t. \( W_\ell \).
4. `torch.clamp(s - cap, min=0.0) ** 2` — the hinge \( \max(0, \sigma - c)^2 \). `torch.clamp` with only `min` acts as ReLU-like; its gradient is 0 below the cap and 1 above, so `**2` gives gradient \( 2(\sigma - c) \) above.
5. `torch.stack` then `.sum()` to combine into one scalar (`torch.stack` is needed because each `sigma` is a 0-dim tensor; a plain `sum(...)` over a list would also work but stacking keeps everything on the graph cleanly).
6. Multiply by \( \lambda \).

### `slug` — reproducibility in the filesystem

```60:64:SIREN_Vista/variants/soft_lipschitz.py
    @staticmethod
    def slug(args):
        return (f"softlip_c{args.soft_lip_cap:g}"
                f"_lam{args.soft_lip_lambda:.0e}"
                f"_{args.soft_lip_apply_to}")
```

Produces strings like `softlip_c1_lam1e-02_sine_only`. Two formatting subtleties:

- `:g` strips trailing zeros: `1.0` → `"1"`, `0.5` → `"0.5"`.
- `:.0e` gives scientific with 0 digits after the point: `0.01` → `"1e-02"`, which is easier to sort and parse than `0.01` in filenames.

The trainer uses this to pick `model_mnist/softlip_c1_lam1e-02_sine_only/modSiren.pth`, so parallel runs with different hyperparams don't clobber each other.

## 2.2 Layer selection

```67:88:SIREN_Vista/variants/soft_lipschitz.py
def _collect_layers(model, mode):
    """Return the list of nn.Linear modules to penalize.

    ``sine_only``         -> SineAffine.affine layers only.
    ``sine_and_readout``  -> + SIREN.hidden2rgb.
    ``all``               -> + ModulatedSIREN(.3D).modul.
    """
    lins = []
    for m in model.modules():
        if isinstance(m, SineAffine):
            lins.append(m.affine)

    if mode in ("sine_and_readout", "all"):
        siren = getattr(model, "siren", None)
        if siren is not None and hasattr(siren, "hidden2rgb"):
            lins.append(siren.hidden2rgb)

    if mode == "all":
        if isinstance(model, (ModulatedSIREN, ModulatedSIREN3D)):
            lins.append(model.modul)

    return lins
```

To understand this, recall the architecture (look at `SIREN_Vista/SIREN.py`):

- `ModulatedSIREN` contains:
  - `self.siren` — a `SIREN` (which is `nn.Sequential` of `SineAffine` layers + `hidden2rgb`).
  - `self.modul` — `nn.Linear(modul_features, hidden_features * num_layers)` mapping \( \phi \) to all the per-layer shifts.

Each `SineAffine` has an `.affine` `nn.Linear` inside:

```41:41:SIREN_Vista/SIREN.py
        self.affine = nn.Linear(in_features, out_features, bias=True)
```

So `_collect_layers` walks `model.modules()` and picks:

| mode | which linears | math impact |
|---|---|---|
| `sine_only` | `SineAffine.affine` for all \( L \) layers | Controls the \( \prod \omega_0 \sigma_\ell \) term, the dominant factor. |
| `sine_and_readout` | above + `siren.hidden2rgb` | Also caps the final projection to the pixel/voxel space. |
| `all` | above + `modul` | Caps the modulation → shift map too. Relevant because the shifts enter inside \( \sin \), so a huge `modul` norm amplifies the effect of \( \phi \). |

The `getattr(model, "siren", None)` is defensive — it works for both `ModulatedSIREN` and `ModulatedSIREN3D`.

Why `sine_only` by default? Because the bound is multiplicative and the sine layers have the \( \omega_0 = 30 \) factor, they dominate. If you can control them you've already won most of the battle. `all` gives a tighter bound but at the cost of more regularization pressure on `modul`, which you may not want (it's the path from the "latent code" into the network and may need to remain expressive).

## 2.3 The power iteration — `_update_uv`

```91:117:SIREN_Vista/variants/soft_lipschitz.py
@torch.no_grad()
def _update_uv(lin, n_iter):
    """One or more power-iteration steps updating the persistent u, v buffers
    attached to ``lin``. Buffers are lazily created on first call.
    """
    W = lin.weight
    W2d = W.reshape(W.shape[0], -1)
    out_dim, in_dim = W2d.shape

    if not hasattr(lin, "_sl_u") or lin._sl_u.shape[0] != out_dim \
            or lin._sl_u.device != W.device or lin._sl_u.dtype != W.dtype:
        u = torch.randn(out_dim, device=W.device, dtype=W.dtype)
        u = u / (u.norm() + 1e-12)
        v = torch.randn(in_dim, device=W.device, dtype=W.dtype)
        v = v / (v.norm() + 1e-12)
        lin._sl_u = u
        lin._sl_v = v

    u = lin._sl_u
    v = lin._sl_v
    for _ in range(max(1, n_iter)):
        v = W2d.t() @ u
        v = v / (v.norm() + 1e-12)
        u = W2d @ v
        u = u / (u.norm() + 1e-12)
    lin._sl_u = u
    lin._sl_v = v
```

Key lines:

**Line 91 — `@torch.no_grad()`.** The entire function runs with autograd off. This is critical because otherwise the computation graph would accumulate across every batch into a monster.

**Line 97 — `W2d = W.reshape(W.shape[0], -1)`.** For `nn.Linear` the weight is already 2D, so this is a no-op. But the same helper could be reused for conv layers where the weight is 4D and you'd want to flatten the input dims. Defensive generality.

**Lines 100–107 — lazy init.** The buffers `_sl_u`, `_sl_v` are attached **to the Linear module itself** (as plain attributes, not `nn.Parameter`), so they persist across calls. The guard checks:
- Buffers exist at all.
- Shape matches (protects against layer resizing — unlikely but free to check).
- Device/dtype match (protects against `.cuda()`/`.cpu()`/autocast moves).

If any check fails, re-initialize. `+ 1e-12` prevents a division-by-zero if the random draw lands near zero.

**Lines 111–115 — the iteration.** One step is:

\[
v \leftarrow \frac{W^\top u}{\|W^\top u\|}, \qquad u \leftarrow \frac{W v}{\|W v\|}.
\]

This is equivalent to one step of power iteration on \( W W^\top \) for `u`, interleaved with one on \( W^\top W \) for `v`. Converges geometrically to the top singular triple \( (\sigma_1, u_1, v_1) \).

**Lines 116–117 — write back.** Persist the updated vectors on the layer for the next batch. This is the warm-start that makes `n_iter=1` sufficient once training is running.

## 2.4 The differentiable singular value — `_power_iter_sigma`

```120:131:SIREN_Vista/variants/soft_lipschitz.py
def _power_iter_sigma(lin, n_iter=1):
    """Estimate the top singular value of ``lin.weight`` differentiably.

    Power iteration updates u, v under no-grad. Then sigma = u^T W v is
    computed with gradients w.r.t. W enabled.
    """
    _update_uv(lin, n_iter)
    W = lin.weight
    W2d = W.reshape(W.shape[0], -1)
    u = lin._sl_u
    v = lin._sl_v
    return torch.dot(u, W2d @ v)
```

**Notice what's missing: no `@torch.no_grad()`.** This function runs under the default grad mode. The sequence:

1. `_update_uv(lin, n_iter)` — refreshes `u`, `v` with grad disabled. They're now constants in the autograd sense.
2. `W = lin.weight` — this is a leaf tensor with `requires_grad=True` (it's an `nn.Parameter`).
3. `W2d @ v` — matrix-vector product. Since `W` has grad and `v` doesn't, the output has a gradient w.r.t. `W`.
4. `torch.dot(u, W2d @ v)` — final scalar, still on the graph, \( \sigma = u^\top W v \).

Running `.backward()` on this scalar (after it's been accumulated into the outer loss) produces

\[
\frac{\partial \sigma}{\partial W} = u v^\top,
\]

which PyTorch computes automatically via the chain rule without any special help from us.

This is why the smoke test from earlier could confirm that after backward, every `SineAffine.affine.weight.grad` was non-zero — the gradient signal really does flow into the weights.

---

# Part 3 — Putting it in motion

Per training batch:

1. Trainer computes reconstruction MSE → `outer_loss` (scalar tensor with grad).
2. Trainer calls `penalty_fn(model)` → `SoftLipschitz.penalty(model, args)`:
   a. `_collect_layers` grabs the sine layers.
   b. For each, `_power_iter_sigma` does: (i) one power-iter step (no grad) refreshing `_sl_u`/`_sl_v`, (ii) returns `torch.dot(u, W v)` (with grad).
   c. `max(0, σ - c)**2` per layer, summed, times λ → scalar \( \Omega \) with grad w.r.t. every penalized `W_ℓ`.
3. `outer_loss = outer_loss + Ω`.
4. `outer_loss.backward()` — gradients of both reconstruction and penalty accumulate into parameters.
5. `clip_grad_norm_`, `optimizer.step()`. Weights update; next batch the power iteration warm-starts from the slightly-stale `_sl_u`, `_sl_v` and re-converges in one step.

After many batches, for layers where the reconstruction objective "wanted" \( \sigma_ℓ \) above \( c \), the optimizer settles at a trade-off where

\[
\underbrace{\frac{\partial \text{MSE}}{\partial W_ℓ}}_{\text{wants large }\sigma_ℓ} \;+\; \underbrace{2\lambda (\sigma_ℓ - c)\, u_ℓ v_ℓ^\top}_{\text{wants }\sigma_ℓ \le c} \;\approx\; 0.
\]

Tune \( \lambda \) up until the spectral norms stay near \( c \) without destroying reconstruction quality. That's the whole experiment.

---

If you want, I can next explain (a) the envelope-theorem argument in more detail, (b) what the likely failure modes of this penalty are empirically (shift-absorption, \( \omega_0 \)-saturation), or (c) how to verify numerically during training that the bound \( \prod \omega_0 \sigma_\ell \) is actually shrinking.