"""Hard SVD spectral projection for SIREN / Spatial-Functa training.

This implements *projected* gradient descent for the spectral-norm ball,
as opposed to the soft penalties in ``variants/soft_lipschitz.py`` and
``variants/spectral_cap.py``. After each outer optimizer step we clamp the
singular values of selected weight matrices so that, exactly,

    sigma_max(W) <= cap.

For a weight ``W = U @ diag(S) @ Vh`` we clamp ``S' = min(S, cap)`` and
rebuild ``W' = U @ diag(S') @ Vh``. Because the singular values of ``W'`` are
exactly ``S'``, the post-projection top singular value is ``min(S[0], cap)``.

SIREN frequency detail
----------------------
A SIREN sine layer ``h = sin(omega_0 (W x + b + shift))`` has local Jacobian
``diag(cos(.)) * omega_0 * W``, so its Lipschitz factor is bounded by
``omega_0 * sigma_max(W)``. To enforce an *effective* Lipschitz cap ``L`` on a
sine layer the raw-matrix cap must be ``L / omega_0``. The readout and
modulation linear maps have no sine frequency, so their caps are used directly.

The module is variant-agnostic: it can be combined with any ``--variant``
(including ``vanilla``) and is fully disabled unless ``--svd-proj`` is set.
"""

import os
from collections import namedtuple

import torch

from SIREN import SineAffine, FinerAffine


# Supported projection scopes.
TARGETS = (
    "readout",
    "pre_readout",
    "readout_and_pre_readout",
    "all_sine_readout",
    "all_sine_readout_modul",
    "modul",
    "modul_readout",
)

CAP_MODES = ("absolute", "reference_scale")

# A single resolved projection target: which Linear to project and the raw cap.
ProjLayer = namedtuple("ProjLayer", ["name", "state_key", "linear", "cap"])


@torch.no_grad()
def project_weight_svd_(weight, cap):
    """Project a weight tensor onto the spectral-norm ball, in place.

    Clamps every singular value of ``weight`` to at most ``cap`` and writes the
    reconstructed matrix back into ``weight``. Only the weight matrix is
    touched; biases are never passed here.

    Args:
        weight: weight tensor, usually shape ``[out_features, in_features]``.
            Higher-rank tensors are reshaped to 2D as ``[out, -1]``.
        cap: maximum allowed spectral norm (top singular value). Must be > 0.

    Returns:
        The pre-projection top singular value (float).
    """
    if cap is None or cap <= 0:
        raise ValueError(f"cap must be a positive number, got {cap!r}")

    original_shape = weight.shape
    # SVD in float32 for numerical stability; copy back in the original dtype.
    W = weight.detach().float().reshape(original_shape[0], -1)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    sigma_before = float(S[0].item())

    S_clamped = torch.clamp(S, max=float(cap))
    W_proj = (U * S_clamped.unsqueeze(0)) @ Vh

    weight.copy_(W_proj.to(dtype=weight.dtype).reshape(original_shape))
    return sigma_before


def _exact_sigma(weight):
    """Exact top singular value of a (possibly >2D) weight tensor."""
    W = weight.detach().float().reshape(weight.shape[0], -1)
    return float(torch.linalg.svdvals(W)[0].item())


def _sine_affine_modules(model):
    """Ordered list of SIREN/FINER affine-carrying layers (the sine layers)."""
    siren = getattr(model, "siren", None)
    layers = []
    if siren is not None and hasattr(siren, "net"):
        for m in siren.net:
            if isinstance(m, (SineAffine, FinerAffine)):
                layers.append(m)
    if not layers:
        layers = [m for m in model.modules() if isinstance(m, (SineAffine, FinerAffine))]
    if not layers:
        raise ValueError("svd projection could not find SIREN/FINER affine layers")
    return layers


def _layer_specs(model, target):
    """Resolve ``target`` into ordered specs.

    Each spec is ``(name, state_key, linear, freq)`` where ``freq`` is the
    sine angular frequency to divide the effective cap by (``None`` for layers
    that use the cap directly, i.e. readout and modul).
    """
    sine = _sine_affine_modules(model)
    siren = getattr(model, "siren", None)
    if siren is None or not hasattr(siren, "hidden2rgb"):
        raise ValueError("svd projection could not find siren.hidden2rgb")
    readout = siren.hidden2rgb
    modul = getattr(model, "modul", None)

    # Index of each sine layer inside siren.net (matches state_dict keys).
    net_index = {}
    if hasattr(siren, "net"):
        for i, m in enumerate(siren.net):
            net_index[id(m)] = i
    last_sine = sine[-1]
    last_idx = net_index.get(id(last_sine), len(sine) - 1)

    def sine_spec(i, m):
        idx = net_index.get(id(m), i)
        freq = float(getattr(m, "freq", getattr(siren, "freq", 30.0)))
        return (f"sine.{i}", f"siren.net.{idx}.affine.weight", m.affine, freq)

    readout_spec = ("readout", "siren.hidden2rgb.weight", readout, None)
    pre_readout_spec = (
        "pre_readout",
        f"siren.net.{last_idx}.affine.weight",
        last_sine.affine,
        float(getattr(last_sine, "freq", getattr(siren, "freq", 30.0))),
    )

    if target == "readout":
        return [readout_spec]
    if target == "pre_readout":
        return [pre_readout_spec]
    if target == "readout_and_pre_readout":
        return [pre_readout_spec, readout_spec]
    if target == "all_sine_readout":
        specs = [sine_spec(i, m) for i, m in enumerate(sine)]
        specs.append(readout_spec)
        return specs
    if target == "all_sine_readout_modul":
        specs = [sine_spec(i, m) for i, m in enumerate(sine)]
        specs.append(readout_spec)
        if modul is None:
            raise ValueError("target 'all_sine_readout_modul' but model has no .modul")
        specs.append(("modul", "modul.weight", modul, None))
        return specs
    if target == "modul":
        if modul is None:
            raise ValueError("target 'modul' but model has no .modul")
        return [("modul", "modul.weight", modul, None)]
    if target == "modul_readout":
        if modul is None:
            raise ValueError("target 'modul_readout' but model has no .modul")
        return [("modul", "modul.weight", modul, None), readout_spec]
    raise ValueError(f"unknown svd projection target {target!r}")


def _effective_cap_for(name, args):
    """Pick the effective Lipschitz cap for a layer group in absolute mode."""
    sine_cap = getattr(args, "svd_proj_sine_cap", None)
    readout_cap = getattr(args, "svd_proj_readout_cap", None)
    modul_cap = getattr(args, "svd_proj_modul_cap", None)
    default = args.svd_proj_cap

    if name == "readout":
        return readout_cap if readout_cap is not None else default
    if name == "modul":
        return modul_cap if modul_cap is not None else default
    # sine.* and pre_readout
    return sine_cap if sine_cap is not None else default


def _reference_raw_sigmas(reference_checkpoint):
    """Exact top singular values of every projectable weight in a checkpoint."""
    ckpt = torch.load(reference_checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    sigmas = {}
    for key, tensor in state.items():
        if not key.endswith(".weight"):
            continue
        if key.startswith("siren.net.") and key.endswith(".affine.weight"):
            sigmas[key] = _exact_sigma(tensor)
        elif key == "siren.hidden2rgb.weight":
            sigmas[key] = _exact_sigma(tensor)
        elif key == "modul.weight":
            sigmas[key] = _exact_sigma(tensor)
    return sigmas


def build_projection_plan(model, args):
    """Build the list of ``ProjLayer`` to project after each optimizer step.

    Returns ``None`` when projection is disabled, so callers can cheaply skip.
    """
    if not getattr(args, "svd_proj", False):
        return None

    specs = _layer_specs(model, args.svd_proj_target)
    mode = args.svd_proj_cap_mode
    freq_adjust = bool(getattr(args, "svd_proj_sine_freq_adjust", True))
    plan = []

    if mode == "absolute":
        for name, key, lin, freq in specs:
            eff = _effective_cap_for(name, args)
            if eff is None or eff <= 0:
                raise ValueError(
                    f"absolute mode needs a positive cap for '{name}'. "
                    "Set --svd-proj-cap (and optional per-group overrides)."
                )
            raw = float(eff)
            if freq is not None and freq_adjust and freq > 0:
                raw = float(eff) / float(freq)
            plan.append(ProjLayer(name, key, lin, raw))
    elif mode == "reference_scale":
        if not args.svd_proj_reference_checkpoint:
            raise ValueError(
                "reference_scale mode needs --svd-proj-reference-checkpoint."
            )
        ref = _reference_raw_sigmas(args.svd_proj_reference_checkpoint)
        scale = float(args.svd_proj_scale)
        for name, key, lin, _freq in specs:
            if key not in ref:
                raise ValueError(
                    f"reference checkpoint missing weight '{key}' for '{name}'."
                )
            plan.append(ProjLayer(name, key, lin, ref[key] * scale))
    else:
        raise ValueError(f"unknown svd projection cap mode {mode!r}")

    return plan


@torch.no_grad()
def apply_projection_(plan):
    """Project every layer in ``plan`` in place.

    Returns a dict ``{name: {cap, sigma_before, sigma_after, state_key}}``.
    ``sigma_after`` is exact: clamping the singular values to ``cap`` makes the
    new top singular value exactly ``min(sigma_before, cap)``.
    """
    stats = {}
    for p in plan:
        before = project_weight_svd_(p.linear.weight, p.cap)
        stats[p.name] = {
            "state_key": p.state_key,
            "cap": float(p.cap),
            "sigma_before": float(before),
            "sigma_after": float(min(before, p.cap)),
        }
    return stats


def resolved_caps(plan):
    """Serializable ``{state_key: cap}`` for checkpoint metadata / verification."""
    if not plan:
        return {}
    return {p.state_key: float(p.cap) for p in plan}


def projection_metadata(args, plan):
    """Compact, serializable description of the projection config for saving."""
    if not getattr(args, "svd_proj", False) or plan is None:
        return None
    return {
        "enabled": True,
        "target": args.svd_proj_target,
        "cap_mode": args.svd_proj_cap_mode,
        "cap": getattr(args, "svd_proj_cap", None),
        "sine_cap": getattr(args, "svd_proj_sine_cap", None),
        "readout_cap": getattr(args, "svd_proj_readout_cap", None),
        "modul_cap": getattr(args, "svd_proj_modul_cap", None),
        "scale": getattr(args, "svd_proj_scale", None),
        "reference_checkpoint": getattr(args, "svd_proj_reference_checkpoint", None),
        "sine_freq_adjust": bool(getattr(args, "svd_proj_sine_freq_adjust", True)),
        "every": int(getattr(args, "svd_proj_every", 1)),
        "resolved_caps": resolved_caps(plan),
    }


def add_args(parser):
    """Register the hard SVD projection CLI flags (style matches trainer.py)."""
    g = parser.add_argument_group("svd_projection")
    g.add_argument("--svd-proj", action="store_true", default=False,
                   help="Enable hard SVD spectral projection after each outer "
                        "optimizer step. Disabled by default (no behavior change).")
    g.add_argument("--svd-proj-target", choices=TARGETS, default="all_sine_readout",
                   help="Which layers to project onto the spectral-norm ball.")
    g.add_argument("--svd-proj-cap-mode", choices=CAP_MODES, default="absolute",
                   help="absolute: caps are effective Lipschitz L values "
                        "(sine layers use L/omega_0). reference_scale: caps are "
                        "scale * reference-checkpoint raw sigma per layer.")
    g.add_argument("--svd-proj-cap", type=float, default=None,
                   help="Default effective Lipschitz cap L for absolute mode. "
                        "Sine/pre_readout caps become L/omega_0; readout and "
                        "modul use L directly.")
    g.add_argument("--svd-proj-sine-cap", type=float, default=None,
                   help="Optional effective cap override for sine layers "
                        "(absolute mode). Falls back to --svd-proj-cap.")
    g.add_argument("--svd-proj-readout-cap", type=float, default=None,
                   help="Optional cap override for the RGB readout "
                        "(absolute mode, used directly). Falls back to --svd-proj-cap.")
    g.add_argument("--svd-proj-modul-cap", type=float, default=None,
                   help="Optional cap override for the modul map "
                        "(absolute mode, used directly). Falls back to --svd-proj-cap.")
    g.add_argument("--svd-proj-scale", type=float, default=0.9,
                   help="Scale for reference_scale mode (e.g. 0.9 = 10%% cut).")
    g.add_argument("--svd-proj-reference-checkpoint", type=str, default=None,
                   help="Reference checkpoint for reference_scale caps.")
    g.add_argument("--svd-proj-sine-freq-adjust", dest="svd_proj_sine_freq_adjust",
                   action="store_true", default=True,
                   help="Divide sine-layer effective caps by omega_0 (default on).")
    g.add_argument("--svd-proj-no-sine-freq-adjust", dest="svd_proj_sine_freq_adjust",
                   action="store_false",
                   help="Do not divide sine-layer caps by omega_0; treat "
                        "--svd-proj-cap as a raw matrix cap for sine layers too.")
    g.add_argument("--svd-proj-every", type=int, default=1,
                   help="Project every N outer optimizer steps (default 1).")


def slug(args):
    """Short tag for run naming, e.g. 'svdproj_all_sine_readout_L1' ."""
    if not getattr(args, "svd_proj", False):
        return ""
    target = args.svd_proj_target
    if args.svd_proj_cap_mode == "reference_scale":
        cap_desc = f"scale{args.svd_proj_scale:g}"
    else:
        cap_desc = f"L{args.svd_proj_cap:g}" if args.svd_proj_cap is not None else "Lmix"
    return f"svdproj_{target}_{cap_desc}"


def verify_checkpoint(checkpoint_path, caps=None, tol=1e-5):
    """Verify projected layers satisfy sigma_max(W) <= cap + tol.

    ``caps`` is ``{state_key: cap}``. When ``None`` it is read from the
    checkpoint's saved ``projection_args['resolved_caps']``.

    Returns ``(ok, rows)`` where ``rows`` is a list of per-layer dicts.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if caps is None:
        proj = ckpt.get("projection_args") or {}
        caps = proj.get("resolved_caps", {})
    if not caps:
        raise ValueError(
            "No caps to verify. The checkpoint has no projection_args; "
            "pass caps={state_key: cap} explicitly."
        )

    rows = []
    ok = True
    for key, cap in caps.items():
        if key not in state:
            rows.append({"layer": key, "cap": cap, "sigma": None,
                         "ok": False, "note": "missing from state_dict"})
            ok = False
            continue
        sigma = _exact_sigma(state[key])
        layer_ok = sigma <= cap + tol
        ok = ok and layer_ok
        rows.append({"layer": key, "cap": float(cap), "sigma": float(sigma),
                     "ok": bool(layer_ok), "note": ""})
    return ok, rows
