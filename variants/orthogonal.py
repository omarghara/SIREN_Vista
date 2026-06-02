"""Orthogonality-regularized SIREN variant.

Adds a training-time penalty that keeps selected linear weights near
orthonormal:

    lambda * mean((G - I)^2)

where G is either W^T W, W W^T, or an automatic choice. For the square hidden
SIREN layers this is exactly the user's requested W^T W - I penalty. The
automatic form uses W^T W when the layer has at least as many rows as columns,
and W W^T for wide layers such as RGB readout where W^T W = I is impossible.
"""

import torch
import torch.nn as nn

from SIREN import SineAffine, FinerAffine, ModulatedSIREN, ModulatedSIREN3D, SpatialModulatedINR

from . import register


@register("orthogonal")
class Orthogonal:
    @staticmethod
    def add_args(parser):
        g = parser.add_argument_group("orthogonal")
        g.add_argument("--orth-lambda", type=float, default=1e-3,
                       help="weight for orthogonality penalty.")
        g.add_argument("--orth-apply-to",
                       choices=["sine_only", "sine_and_readout", "all"],
                       default="sine_only",
                       help="'sine_only' penalizes SIREN/FINER affine layers; "
                            "'sine_and_readout' also penalizes hidden2rgb; "
                            "'all' also penalizes the modulation map.")
        g.add_argument("--orth-form", choices=["auto", "columns", "rows"],
                       default="auto",
                       help="'columns' uses W^T W - I, 'rows' uses W W^T - I, "
                            "and 'auto' uses columns unless rows are required.")
        g.add_argument("--orth-skip-first", action="store_true", default=False,
                       help="exclude the first coordinate-input sine layer.")

    @staticmethod
    def build(base_model, args):
        return base_model

    @staticmethod
    def penalty(model, args):
        layers = _collect_layers(
            model,
            mode=args.orth_apply_to,
            skip_first=args.orth_skip_first,
        )
        if not layers:
            return torch.zeros((), device=next(model.parameters()).device)
        terms = [_orth_penalty(lin.weight, form=args.orth_form) for lin in layers]
        return args.orth_lambda * torch.stack(terms).sum()

    @staticmethod
    def slug(args):
        slug = f"orth_lam{args.orth_lambda:.0e}_{args.orth_apply_to}_{args.orth_form}"
        if getattr(args, "orth_skip_first", False):
            slug += "_skip0"
        return slug


def _collect_layers(model, mode, skip_first=False):
    layers = []
    seen_first = False
    for module in model.modules():
        if isinstance(module, (SineAffine, FinerAffine)):
            if skip_first and not seen_first:
                seen_first = True
                continue
            seen_first = True
            layers.append(module.affine)

    if mode in ("sine_and_readout", "all"):
        siren = getattr(model, "siren", None)
        if siren is not None and hasattr(siren, "hidden2rgb"):
            layers.append(siren.hidden2rgb)

    if mode == "all" and isinstance(model, (ModulatedSIREN, ModulatedSIREN3D, SpatialModulatedINR)):
        modul = getattr(model, "modul", None)
        if isinstance(modul, nn.Linear):
            layers.append(modul)

    return layers


def _orth_penalty(weight, form="auto"):
    W = weight.reshape(weight.shape[0], -1)
    out_dim, in_dim = W.shape

    if form == "auto":
        use_columns = out_dim >= in_dim
    elif form == "columns":
        use_columns = True
    elif form == "rows":
        use_columns = False
    else:
        raise ValueError(f"unknown orthogonality form {form!r}")

    if use_columns:
        gram = W.t() @ W
    else:
        gram = W @ W.t()
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - eye).pow(2).mean()
