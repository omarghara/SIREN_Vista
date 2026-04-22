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

import torch
import torch.nn as nn

from SIREN import SineAffine, ModulatedSIREN, ModulatedSIREN3D

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

    @staticmethod
    def build(base_model, args):
        return base_model

    @staticmethod
    def penalty(model, args):
        pairs = _collect_layers(model, args.soft_lip_apply_to,
                                L=args.soft_lip_cap,
                                skip_first=getattr(args, "soft_lip_skip_first", False))
        if not pairs:
            return torch.zeros((), device=next(model.parameters()).device)
        terms = []
        for lin, cap in pairs:
            sigma = _power_iter_sigma(lin, n_iter=args.soft_lip_power_iters)
            terms.append(torch.clamp(sigma - cap, min=0.0) ** 2)
        return args.soft_lip_lambda * torch.stack(terms).sum()

    @staticmethod
    def slug(args):
        slug = (f"softlip_L{args.soft_lip_cap:g}"
                f"_lam{args.soft_lip_lambda:.0e}"
                f"_{args.soft_lip_apply_to}")
        if getattr(args, "soft_lip_skip_first", False):
            slug += "_skip0"
        return slug


def _collect_layers(model, mode, L, skip_first=False):
    """Return list of ``(nn.Linear, sigma_cap)`` pairs to penalize.

    ``L`` is the per-layer Lipschitz budget. Caps are derived per layer:

    * ``SineAffine.affine``         -> cap = L / layer.freq
    * ``SIREN.hidden2rgb``          -> cap = L
    * ``ModulatedSIREN(3D).modul``  -> cap = L / sine_freq
      (fallback 30.0 if no SineAffine is found to read ``freq`` from).

    ``sine_only``        -> SineAffine.affine layers only.
    ``sine_and_readout`` -> + SIREN.hidden2rgb.
    ``all``              -> + ModulatedSIREN(.3D).modul.

    If ``skip_first`` is True the first ``SineAffine`` encountered (the
    coordinate-input layer, sine.0) is omitted from the returned list.
    ``sine_freq`` is still read from it so downstream cap derivations are
    unchanged.
    """
    pairs = []
    sine_freq = None
    seen_first_sine = False
    for m in model.modules():
        if isinstance(m, SineAffine):
            if sine_freq is None:
                sine_freq = m.freq
            if skip_first and not seen_first_sine:
                seen_first_sine = True
                continue
            seen_first_sine = True
            pairs.append((m.affine, L / m.freq))

    if mode in ("sine_and_readout", "all"):
        siren = getattr(model, "siren", None)
        if siren is not None and hasattr(siren, "hidden2rgb"):
            pairs.append((siren.hidden2rgb, L))

    if mode == "all":
        if isinstance(model, (ModulatedSIREN, ModulatedSIREN3D)):
            modul_freq = sine_freq if sine_freq is not None else 30.0
            pairs.append((model.modul, L / modul_freq))

    return pairs


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
