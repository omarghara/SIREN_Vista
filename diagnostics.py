"""Lightweight training diagnostics for the SIREN backbone.

Currently exposes one helper: ``layer_sigmas(model)``, which returns a
``{layer_name: sigma_top}`` mapping for the interesting linear layers of a
``ModulatedSIREN`` / ``ModulatedSIREN3D``. Spectral norms are estimated with
fresh-buffer power iteration (no dependency on ``torch.nn.utils.spectral_norm``
or the ``soft_lipschitz`` variant's persistent buffers). Intended for periodic
logging during training or for post-hoc inspection of trained checkpoints.
"""

from collections import OrderedDict

import torch

from SIREN import ModulatedSIREN, ModulatedSIREN3D, SineAffine


@torch.no_grad()
def _power_iter_sigma(W2d, n_iter=30, eps=1e-12):
    """Fresh-buffer estimate of the top singular value of a 2D matrix.

    Not differentiable. Uses random init + ``n_iter`` power iterations. Use
    ``n_iter >= 30`` to get reliable values when called ad-hoc (as opposed
    to the warm-started single-step pattern used inside soft-Lipschitz).
    """
    out_dim, in_dim = W2d.shape
    device, dtype = W2d.device, W2d.dtype
    u = torch.randn(out_dim, device=device, dtype=dtype)
    u = u / (u.norm() + eps)
    v = torch.randn(in_dim, device=device, dtype=dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iter):
        v = W2d.t() @ u
        v = v / (v.norm() + eps)
        u = W2d @ v
        u = u / (u.norm() + eps)
    return float(torch.dot(u, W2d @ v).item())


@torch.no_grad()
def layer_sigmas(model, n_iter=30):
    """Return an ``OrderedDict`` ``{name: sigma_top}`` for the interesting
    linear layers of a ``ModulatedSIREN`` / ``ModulatedSIREN3D``.

    Grouping mirrors ``variants/soft_lipschitz.py::_collect_layers``:

    * ``sine.{k}`` — one per ``SineAffine.affine`` in forward order
    * ``readout`` — ``SIREN.hidden2rgb`` (only if present)
    * ``modul``   — ``ModulatedSIREN.modul`` (only if present)
    """
    out = OrderedDict()
    sine_idx = 0
    for m in model.modules():
        if isinstance(m, SineAffine):
            W = m.affine.weight
            out[f'sine.{sine_idx}'] = _power_iter_sigma(
                W.reshape(W.shape[0], -1), n_iter
            )
            sine_idx += 1

    if isinstance(model, (ModulatedSIREN, ModulatedSIREN3D)):
        siren = getattr(model, 'siren', None)
        if siren is not None and hasattr(siren, 'hidden2rgb'):
            W = siren.hidden2rgb.weight
            out['readout'] = _power_iter_sigma(
                W.reshape(W.shape[0], -1), n_iter
            )
        if hasattr(model, 'modul'):
            W = model.modul.weight
            out['modul'] = _power_iter_sigma(
                W.reshape(W.shape[0], -1), n_iter
            )
    return out


def format_sigmas_one_liner(sigmas):
    """Compact one-line formatter suitable for tqdm.write / print inside a
    training loop.

    Examples:
        sine: [0.710 0.121 0.109 ...] min=0.101 max=0.714 | readout: 0.080 | modul: 1.832
    """
    sine_items = [(k, v) for k, v in sigmas.items() if k.startswith('sine.')]
    parts = []
    if sine_items:
        vals = [v for _, v in sine_items]
        if len(vals) <= 5:
            parts.append('sine: [' + ' '.join(f'{v:.3f}' for v in vals) + ']')
        else:
            head = ' '.join(f'{v:.3f}' for v in vals[:3])
            parts.append(
                f'sine[{len(vals)}]: [{head} ...] '
                f'min={min(vals):.3f} max={max(vals):.3f}'
            )
    if 'readout' in sigmas:
        parts.append(f'readout: {sigmas["readout"]:.3f}')
    if 'modul' in sigmas:
        parts.append(f'modul: {sigmas["modul"]:.3f}')
    return ' | '.join(parts)
