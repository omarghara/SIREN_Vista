"""Soft-Lipschitz SIREN variant.

Adds a training-time penalty
``lambda * sum_l max(0, sigma(W_l) - c_l)^2`` over selected linear layers.
The per-layer spectral-norm cap ``c_l`` is derived from a single per-layer
Lipschitz budget ``L = args.soft_lip_cap``:

* Sine hidden layers (``SineAffine.affine``) -> ``c_l = L / freq``.
  The layer map is ``sin(freq * (W x + b + s))`` with Lipschitz
  ``freq * sigma(W)``; capping ``sigma(W)`` at ``L/freq`` bounds it by ``L``.
* Readout (``SIREN.hidden2rgb``)             -> ``c_l = L``.
  Pure linear readout, no sine, so its Lipschitz is ``sigma(W)``.
* Modulation (``modul``, mode 'all')         -> ``c_l = L / freq``.
  The shift it produces feeds into a sine, so it is also amplified by freq.

The first sine layer (``siren.net[0]``, a.k.a. ``sine.0``) can optionally
be excluded via ``--soft-lip-skip-first``. Rationale: sigma(W_0) does not
appear in the phi -> output Lipschitz bound (only cos(omega_0*(W_0 x + ...))
factors do, and those are bounded by 1 regardless of W_0), so penalizing
it does not tighten the certificate. Its coordinate-input init also
produces a naturally large sigma (~5) that would otherwise dominate the
penalty budget.

Spectral norms are estimated via power iteration with persistent u/v
buffers (same pattern as ``torch.nn.utils.spectral_norm``, but we do not
rescale the weights). No architectural change: ``build`` is identity. The
penalty flows gradients back into each penalized ``W_l`` because
``sigma = u^T W v`` is differentiable w.r.t. ``W`` while ``u``, ``v`` are
updated under ``torch.no_grad()``.
"""

import os

import torch
import torch.nn as nn

from SIREN import SineAffine, FinerAffine, ModulatedSIREN, ModulatedSIREN3D

from . import register


@register("soft_lipschitz")
class SoftLipschitz:
    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group("soft_lipschitz")
        g.add_argument("--soft-lip-cap", type=float, default=1.0,
                       help="per-layer Lipschitz budget L. "
                            "Sine layers get spectral-norm cap L/freq so "
                            "that freq*sigma(W) <= L; the linear readout "
                            "hidden2rgb gets cap L directly.")
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
        g.add_argument("--soft-lip-skip-first", action="store_true",
                       help="exclude the first sine layer (sine.0) from the "
                            "penalty. sigma(W_0) does not enter the "
                            "phi->output Lipschitz bound, so penalizing it "
                            "cannot tighten the certificate; its large "
                            "coord-input init (~5) also otherwise dominates "
                            "the penalty budget.")
        g.add_argument("--soft-lip-reference-checkpoint", type=str, default=None,
                       help="Optional checkpoint whose layer spectral norms are "
                            "used as per-layer caps after multiplying by "
                            "--soft-lip-reference-scale. This is useful for "
                            "FINER/SIREN spatial runs where caps should be "
                            "derived from a specific baseline backbone.")
        g.add_argument("--soft-lip-reference-scale", type=float, default=0.90,
                       help="Multiplier applied to reference checkpoint sigmas "
                            "when --soft-lip-reference-checkpoint is set.")

    @staticmethod
    def build(base_model, args):
        return base_model

    @staticmethod
    def penalty(model, args):
        pairs = _collect_layers(model, args)
        if not pairs:
            return torch.zeros((), device=next(model.parameters()).device)
        terms = []
        for lin, cap in pairs:
            sigma = _power_iter_sigma(lin, n_iter=args.soft_lip_power_iters)
            terms.append(torch.clamp(sigma - cap, min=0.0) ** 2)
        return args.soft_lip_lambda * torch.stack(terms).sum()

    @staticmethod
    def slug(args):
        if getattr(args, "soft_lip_reference_checkpoint", None):
            ref = os.path.basename(os.path.dirname(args.soft_lip_reference_checkpoint))
            slug = (f"softlip_ref{ref[:16]}"
                    f"_scale{args.soft_lip_reference_scale:g}"
                    f"_lam{args.soft_lip_lambda:.0e}"
                    f"_{args.soft_lip_apply_to}")
        else:
            slug = (f"softlip_L{args.soft_lip_cap:g}"
                    f"_lam{args.soft_lip_lambda:.0e}"
                    f"_{args.soft_lip_apply_to}")
        if getattr(args, "soft_lip_skip_first", False):
            slug += "_skip0"
        return slug


# def _collect_layers(model, mode, L, skip_first=False):
#     """Return list of ``(nn.Linear, sigma_cap)`` pairs to penalize.

#     ``L`` is the per-layer Lipschitz budget. Caps are derived per layer:

#     * ``SineAffine.affine``         -> cap = L / layer.freq
#     * ``SIREN.hidden2rgb``          -> cap = L
#     * ``ModulatedSIREN(3D).modul``  -> cap = L / sine_freq
#       (fallback 30.0 if no SineAffine is found to read ``freq`` from).

#     ``sine_only``        -> SineAffine.affine layers only.
#     ``sine_and_readout`` -> + SIREN.hidden2rgb.
#     ``all``              -> + ModulatedSIREN(.3D).modul.

#     If ``skip_first`` is True the first ``SineAffine`` encountered (the
#     coordinate-input layer, sine.0) is omitted from the returned list.
#     ``sine_freq`` is still read from it so downstream cap derivations are
#     unchanged.
#     """
#     pairs = []
#     sine_freq = None
#     seen_first_sine = False
#     for m in model.modules():
#         if isinstance(m, SineAffine):
#             if sine_freq is None:
#                 sine_freq = m.freq
#             if skip_first and not seen_first_sine:
#                 pairs.append((m.affine, 4 / m.freq))

#                 seen_first_sine = True
#                 continue
#             seen_first_sine = True
#             pairs.append((m.affine, L / m.freq))

#     if mode in ("sine_and_readout", "all"):
#         siren = getattr(model, "siren", None)
#         if siren is not None and hasattr(siren, "hidden2rgb"):
#             pairs.append((siren.hidden2rgb, L))

#     if mode == "all":
#         if isinstance(model, (ModulatedSIREN, ModulatedSIREN3D)):
#             modul_freq = sine_freq if sine_freq is not None else 30.0
#             pairs.append((model.modul, L / modul_freq))

#     return pairs

# 90 harcap experiment
# def _collect_layers(model, mode, L, skip_first=False):
#     """Return list of (nn.Linear, sigma_cap) pairs to penalize.

#     HARD-CODED EXPERIMENT:
#     This version ignores the global L for SIREN layers and instead uses
#     per-layer spectral-norm caps derived from the vanilla MNIST model.

#     Goal:
#         Reduce each vanilla layer spectral bound by about 10%.

#     Important:
#         - We penalize only SIREN weights:
#             sine.0, sine.1, ..., sine.9, readout
#         - We do NOT penalize the modulation matrix in this experiment.
#         - skip_first is ignored for now because we explicitly define sine.0 cap.

#     Vanilla measured sigma_1 values:
#         sine.0  = 4.982442
#         sine.1  = 0.092663
#         sine.2  = 0.094855
#         sine.3  = 0.092998
#         sine.4  = 0.093318
#         sine.5  = 0.097486
#         sine.6  = 0.105713
#         sine.7  = 0.119848
#         sine.8  = 0.124839
#         sine.9  = 0.125309
#         readout = 0.061992

#     Caps below are 90% of the vanilla values.
#     This is a mild 10% reduction target.
#     """

#     # 10% reduction from vanilla sigma_1 values.
#     # These are RAW spectral norm caps, not effective freq*sigma caps.
#     hardcoded_sigma_caps = {
#         "sine.0": 4.982442 * 0.90,
#         "sine.1": 0.092663 * 0.90,
#         "sine.2": 0.094855 * 0.90,
#         "sine.3": 0.092998 * 0.90,
#         "sine.4": 0.093318 * 0.90,
#         "sine.5": 0.097486 * 0.90,
#         "sine.6": 0.105713 * 0.90,
#         "sine.7": 0.119848 * 0.90,
#         "sine.8": 0.124839 * 0.90,
#         "sine.9": 0.125309 * 0.90,
#         "readout": 0.061992 * 0.90,
#     }

#     pairs = []

#     sine_idx = 0
#     for m in model.modules():
#         if isinstance(m, SineAffine):
#             layer_name = f"sine.{sine_idx}"

#             if layer_name in hardcoded_sigma_caps:
#                 pairs.append((m.affine, hardcoded_sigma_caps[layer_name]))

#             sine_idx += 1

#     # Penalize readout because it is a SIREN weight.
#     # Do not penalize modul in this experiment.
#     if mode in ("sine_and_readout", "all"):
#         siren = getattr(model, "siren", None)
#         if siren is not None and hasattr(siren, "hidden2rgb"):
#             pairs.append((siren.hidden2rgb, hardcoded_sigma_caps["readout"]))

#     return pairs


_REFERENCE_CAP_CACHE = {}


def _collect_layers(model, args):
    reference_checkpoint = getattr(args, "soft_lip_reference_checkpoint", None)
    if reference_checkpoint:
        return _collect_reference_scaled_layers(
            model,
            mode=args.soft_lip_apply_to,
            reference_checkpoint=reference_checkpoint,
            scale=float(getattr(args, "soft_lip_reference_scale", 0.90)),
            skip_first=getattr(args, "soft_lip_skip_first", False),
        )

    return _collect_hardcoded_layers(
        model,
        mode=args.soft_lip_apply_to,
        skip_first=getattr(args, "soft_lip_skip_first", False),
    )


def _collect_hardcoded_layers(model, mode, skip_first=False):
    """Return list of (nn.Linear, sigma_cap) pairs to penalize.

    HARD-CODED EXPERIMENT:
    first95_rest80

    Caps:
        sine.0       = 95% of vanilla sigma_1
        sine.1-9     = 80% of vanilla sigma_1
        readout      = 80% of vanilla sigma_1
        modul        = not capped

    Important:
        - Sine caps are raw sigma(W) caps.
        - Readout cap is raw sigma(W) cap.
        - Modul is intentionally not included.
    """

    vanilla_sigmas = {
        "sine.0": 4.982442,
        "sine.1": 0.092663,
        "sine.2": 0.094855,
        "sine.3": 0.092998,
        "sine.4": 0.093318,
        "sine.5": 0.097486,
        "sine.6": 0.105713,
        "sine.7": 0.119848,
        "sine.8": 0.124839,
        "sine.9": 0.125309,
        "readout": 0.061992,
    }

    hardcoded_sigma_caps = {
        # First coordinate-input sine layer: mild cap.
        "sine.0": vanilla_sigmas["sine.0"] * 0.95,

        # Hidden sine layers: stronger cap.
        "sine.1": vanilla_sigmas["sine.1"] * 0.80,
        "sine.2": vanilla_sigmas["sine.2"] * 0.80,
        "sine.3": vanilla_sigmas["sine.3"] * 0.80,
        "sine.4": vanilla_sigmas["sine.4"] * 0.80,
        "sine.5": vanilla_sigmas["sine.5"] * 0.80,
        "sine.6": vanilla_sigmas["sine.6"] * 0.80,
        "sine.7": vanilla_sigmas["sine.7"] * 0.80,
        "sine.8": vanilla_sigmas["sine.8"] * 0.80,
        "sine.9": vanilla_sigmas["sine.9"] * 0.80,

        # Readout: stronger cap.
        "readout": vanilla_sigmas["readout"] * 0.80,
    }

    pairs = []

    sine_idx = 0
    for m in model.modules():
        if isinstance(m, SineAffine):
            layer_name = f"sine.{sine_idx}"

            if layer_name in hardcoded_sigma_caps:
                pairs.append((m.affine, hardcoded_sigma_caps[layer_name]))

            sine_idx += 1

    if mode in ("sine_and_readout", "all"):
        siren = getattr(model, "siren", None)
        if siren is not None and hasattr(siren, "hidden2rgb"):
            pairs.append((siren.hidden2rgb, hardcoded_sigma_caps["readout"]))

    # Do not cap modul in this experiment.
    return pairs


def _collect_reference_scaled_layers(model, mode, reference_checkpoint, scale, skip_first=False):
    """Cap current model layers by `scale * sigma(reference layer)`.

    Reference checkpoints use stable state_dict names for both SIREN and FINER:

    - `siren.net.{i}.affine.weight`
    - `siren.hidden2rgb.weight`
    - `modul.weight` (only when `mode == "all"`)

    Current layers are matched by order. This supports SIREN (`SineAffine`) and
    FINER (`FinerAffine`) spatial/global backbones.
    """
    cache_key = (os.path.abspath(reference_checkpoint), float(scale), mode, bool(skip_first))
    caps = _REFERENCE_CAP_CACHE.get(cache_key)
    if caps is None:
        caps = _reference_caps_from_checkpoint(
            reference_checkpoint=reference_checkpoint,
            scale=scale,
            mode=mode,
            skip_first=skip_first,
        )
        _REFERENCE_CAP_CACHE[cache_key] = caps

    pairs = []
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, (SineAffine, FinerAffine)):
            cap = caps.get(f"siren.net.{layer_idx}.affine.weight")
            if cap is not None:
                pairs.append((m.affine, cap))
            layer_idx += 1

    if mode in ("sine_and_readout", "all"):
        siren = getattr(model, "siren", None)
        if siren is not None and hasattr(siren, "hidden2rgb"):
            cap = caps.get("siren.hidden2rgb.weight")
            if cap is not None:
                pairs.append((siren.hidden2rgb, cap))

    if mode == "all":
        modul = getattr(model, "modul", None)
        cap = caps.get("modul.weight")
        if modul is not None and cap is not None:
            pairs.append((modul, cap))

    return pairs


def _reference_caps_from_checkpoint(reference_checkpoint, scale, mode, skip_first=False):
    ckpt = torch.load(reference_checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    caps = {}

    layer_idx = 0
    while True:
        key = f"siren.net.{layer_idx}.affine.weight"
        if key not in state:
            break
        if not (skip_first and layer_idx == 0):
            caps[key] = _exact_sigma(state[key]) * scale
        layer_idx += 1

    if mode in ("sine_and_readout", "all") and "siren.hidden2rgb.weight" in state:
        caps["siren.hidden2rgb.weight"] = _exact_sigma(state["siren.hidden2rgb.weight"]) * scale

    if mode == "all" and "modul.weight" in state:
        caps["modul.weight"] = _exact_sigma(state["modul.weight"]) * scale

    return caps


def _exact_sigma(weight):
    W2d = weight.detach().float().reshape(weight.shape[0], -1)
    return float(torch.linalg.svdvals(W2d)[0].item())

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
