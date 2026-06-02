"""Targeted spectral-cap SIREN variant.

This variant is for fast follow-up experiments that warm-start from a trained
vanilla backbone and constrain only a late layer. It adds:

    lambda * sum_l max(0, sigma(W_l) - cap_l)^2

where the target can be the RGB readout, the final sine layer just before the
readout, or both. Caps can be a fixed fraction of a reference checkpoint, or
computed to counter the product Lipschitz upper bound of the earlier SIREN
layers in one chosen late layer.
"""

import os

import torch
import torch.nn as nn

from SIREN import SineAffine, FinerAffine

from . import register


@register("spectral_cap")
class SpectralCap:
    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group("spectral_cap")
        g.add_argument("--spec-cap-lambda", type=float, default=1e-2,
                       help="weight for spectral cap penalty.")
        g.add_argument("--spec-cap-target",
                       choices=["readout", "pre_readout", "readout_and_pre_readout"],
                       default="readout",
                       help="which late layer to cap. pre_readout is the final "
                            "SineAffine/FinerAffine layer before hidden2rgb.")
        g.add_argument("--spec-cap-mode",
                       choices=["reference_scale", "counter_amplification"],
                       default="reference_scale",
                       help="reference_scale caps target layers at scale times "
                            "their reference sigma. counter_amplification sets "
                            "one late-layer cap so the reference product "
                            "Lipschitz upper bound is at most --spec-cap-counter-target.")
        g.add_argument("--spec-cap-reference-checkpoint", type=str, default=None,
                       help="vanilla/reference checkpoint used to compute caps.")
        g.add_argument("--spec-cap-scale", type=float, default=0.9,
                       help="reference scale for reference_scale mode. Use 0.9 "
                            "for a 10 percent cap reduction and 0.5 for a 50 "
                            "percent cap reduction.")
        g.add_argument("--spec-cap-absolute", type=float, default=None,
                       help="optional absolute spectral cap. If set, this "
                            "overrides reference_scale caps.")
        g.add_argument("--spec-cap-counter-target", type=float, default=1.0,
                       help="desired total product-Lipschitz upper bound in "
                            "counter_amplification mode.")
        g.add_argument("--spec-cap-power-iters", type=int, default=10,
                       help="power-iteration steps per batch for sigma estimate.")

    @staticmethod
    def build(base_model, args):
        return base_model

    @staticmethod
    def penalty(model, args):
        pairs = _collect_capped_layers(model, args)
        if not pairs:
            return torch.zeros((), device=next(model.parameters()).device)
        terms = []
        for lin, cap in pairs:
            sigma = _power_iter_sigma(lin, n_iter=args.spec_cap_power_iters)
            terms.append(torch.clamp(sigma - cap, min=0.0).pow(2))
        return args.spec_cap_lambda * torch.stack(terms).sum()

    @staticmethod
    def slug(args):
        target = args.spec_cap_target
        if args.spec_cap_mode == "counter_amplification":
            cap_desc = f"counter{args.spec_cap_counter_target:g}"
        elif args.spec_cap_absolute is not None:
            cap_desc = f"abs{args.spec_cap_absolute:g}"
        else:
            cap_desc = f"scale{args.spec_cap_scale:g}"
        ref_desc = "noref"
        if args.spec_cap_reference_checkpoint:
            ref_desc = os.path.basename(os.path.dirname(args.spec_cap_reference_checkpoint))[:16]
        return f"speccap_{target}_{cap_desc}_lam{args.spec_cap_lambda:.0e}_ref{ref_desc}"


def _collect_capped_layers(model, args):
    reference_checkpoint = args.spec_cap_reference_checkpoint
    if args.spec_cap_absolute is None and reference_checkpoint is None:
        raise ValueError(
            "--spec-cap-reference-checkpoint is required unless "
            "--spec-cap-absolute is provided."
        )

    targets = _target_layers(model, args.spec_cap_target)
    if args.spec_cap_absolute is not None:
        return [(lin, float(args.spec_cap_absolute)) for _, lin in targets]

    ref = _reference_stats(reference_checkpoint)

    if args.spec_cap_mode == "reference_scale":
        caps = {}
        for name, _ in targets:
            key = _target_to_reference_key(name, ref)
            caps[name] = ref["sigmas"][key] * float(args.spec_cap_scale)
        return [(lin, caps[name]) for name, lin in targets]

    if args.spec_cap_mode == "counter_amplification":
        if args.spec_cap_target == "readout_and_pre_readout":
            raise ValueError(
                "counter_amplification supports one target at a time: "
                "use readout or pre_readout."
            )
        cap = _counter_amplification_cap(
            ref,
            target=args.spec_cap_target,
            target_total=float(args.spec_cap_counter_target),
        )
        return [(targets[0][1], cap)]

    raise ValueError(f"unknown spec cap mode {args.spec_cap_mode!r}")


def _target_layers(model, target):
    sine_layers = [m.affine for m in model.modules() if isinstance(m, (SineAffine, FinerAffine))]
    if not sine_layers:
        raise ValueError("spectral_cap could not find SIREN/FINER affine layers")
    siren = getattr(model, "siren", None)
    if siren is None or not hasattr(siren, "hidden2rgb"):
        raise ValueError("spectral_cap could not find siren.hidden2rgb")

    if target == "readout":
        return [("readout", siren.hidden2rgb)]
    if target == "pre_readout":
        return [("pre_readout", sine_layers[-1])]
    if target == "readout_and_pre_readout":
        return [("pre_readout", sine_layers[-1]), ("readout", siren.hidden2rgb)]
    raise ValueError(f"unknown spectral cap target {target!r}")


def _target_to_reference_key(target, ref):
    if target == "readout":
        return "siren.hidden2rgb.weight"
    if target == "pre_readout":
        return ref["sine_keys"][-1]
    raise ValueError(f"unknown target {target!r}")


def _counter_amplification_cap(ref, target, target_total=1.0):
    sigmas = ref["sigmas"]
    sine_keys = ref["sine_keys"]
    freqs = ref["freqs"]
    readout_sigma = sigmas["siren.hidden2rgb.weight"]

    if target == "readout":
        sine_amp = 1.0
        for key, freq in zip(sine_keys, freqs):
            sine_amp *= max(freq * sigmas[key], 1e-12)
        return target_total / max(sine_amp, 1e-12)

    if target == "pre_readout":
        prev_amp = 1.0
        for key, freq in zip(sine_keys[:-1], freqs[:-1]):
            prev_amp *= max(freq * sigmas[key], 1e-12)
        last_freq = freqs[-1]
        denom = max(readout_sigma * prev_amp * last_freq, 1e-12)
        return target_total / denom

    raise ValueError(f"counter amplification unsupported target {target!r}")


def _reference_stats(reference_checkpoint):
    ckpt = torch.load(reference_checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_args = ckpt.get("model_args", {})
    default_freq = float(model_args.get("freq", 30.0))

    sigmas = {}
    sine_keys = []
    idx = 0
    while True:
        key = f"siren.net.{idx}.affine.weight"
        if key not in state:
            break
        sigmas[key] = _exact_sigma(state[key])
        sine_keys.append(key)
        idx += 1

    if "siren.hidden2rgb.weight" not in state:
        raise ValueError(
            f"reference checkpoint {reference_checkpoint!r} has no "
            "siren.hidden2rgb.weight"
        )
    sigmas["siren.hidden2rgb.weight"] = _exact_sigma(state["siren.hidden2rgb.weight"])
    freqs = [default_freq for _ in sine_keys]
    return {"sigmas": sigmas, "sine_keys": sine_keys, "freqs": freqs}


def _exact_sigma(weight):
    W = weight.detach().float().reshape(weight.shape[0], -1)
    return float(torch.linalg.svdvals(W)[0].item())


@torch.no_grad()
def _update_uv(lin, n_iter):
    W = lin.weight
    W2d = W.reshape(W.shape[0], -1)
    out_dim, in_dim = W2d.shape
    if not hasattr(lin, "_sc_u") or lin._sc_u.shape[0] != out_dim \
            or lin._sc_u.device != W.device or lin._sc_u.dtype != W.dtype:
        u = torch.randn(out_dim, device=W.device, dtype=W.dtype)
        u = u / (u.norm() + 1e-12)
        v = torch.randn(in_dim, device=W.device, dtype=W.dtype)
        v = v / (v.norm() + 1e-12)
        lin._sc_u = u
        lin._sc_v = v

    u = lin._sc_u
    v = lin._sc_v
    for _ in range(max(1, n_iter)):
        v = W2d.t() @ u
        v = v / (v.norm() + 1e-12)
        u = W2d @ v
        u = u / (u.norm() + 1e-12)
    lin._sc_u = u
    lin._sc_v = v


def _power_iter_sigma(lin, n_iter=1):
    _update_uv(lin, n_iter)
    W = lin.weight
    W2d = W.reshape(W.shape[0], -1)
    return torch.dot(lin._sc_u, W2d @ lin._sc_v)
