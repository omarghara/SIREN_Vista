"""Reconstruction evaluation for trained SIREN backbones.

Re-fits per-sample modulations on train and/or test splits and reports
MSE / PSNR / SSIM at multiple inner-loop iteration checkpoints, matching
the fitting-curve style used in the Functa and SIREN papers:

  - SIREN (Sitzmann et al. 2020, arXiv:2006.09661): PSNR vs training step.
  - Functa (Dupont et al. 2022, arXiv:2201.12204): PSNR of reconstructed
    image from the fitted functa, reported as a fitting curve.

Unlike makeset.py, this script does not save the functaset; it only
measures reconstruction quality. The inner-loop init, optimizer and lr
mirror makeset.py exactly so the ``iters=5`` row of the reported curve
reproduces what the downstream classifier actually sees when trained on
a ``makeset.py --iters 5`` functaset; the higher-iter rows show the
backbone's expressive ceiling.

The fit is batched: B modulators are optimized in parallel per forward
pass via a local ``batched_forward`` that bypasses ``ModulatedSIREN``'s
per-layer ``shift`` attribute (which is scalar-per-layer, not
batch-per-layer). Because each sample's loss depends only on its own
modulator, sum-over-batch of per-image MSE gives per-sample gradients
identical to running B independent fits. Pick ``--batch-size`` by GPU
memory; 64 is a good MNIST default, 256 is usually still comfortable.
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataloader import get_cifar10_loader, get_mnist_loader
from dataloader_modelnet import get_modelnet_loader
from SIREN import ModulatedFourierSIREN, ModulatedSIREN, ModulatedSIREN3D
from utils import set_random_seeds
import variants


# ---------------------------------------------------------------------------
# SSIM (hand-rolled, data_range = 1.0) -- batched along dim 0.
# ---------------------------------------------------------------------------

def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]  # (1, 1, k, k)


def ssim_2d_batch(fitted, target, window_size=11, sigma=1.5):
    """SSIM for batched 2D images in [0, 1].

    fitted, target: (B, H, W). Returns a (B,) float tensor (one mean-over-
    spatial SSIM per image). F.conv2d batches over dim 0 natively, so one
    call replaces B per-image calls.
    """
    assert fitted.ndim == 3 and target.ndim == 3, \
        f"ssim_2d_batch expects (B, H, W), got {fitted.shape} and {target.shape}"
    window = _gaussian_window(window_size, sigma, device=fitted.device, dtype=fitted.dtype)
    x = fitted.unsqueeze(1)  # (B, 1, H, W)
    y = target.unsqueeze(1)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    sig_x = F.conv2d(x * x, window, padding=pad) - mu_x.pow(2)
    sig_y = F.conv2d(y * y, window, padding=pad) - mu_y.pow(2)
    sig_xy = F.conv2d(x * y, window, padding=pad) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sig_x + sig_y + C2)
    return (num / den).mean(dim=(1, 2, 3))  # (B,)


def ssim_image_batch(fitted, target, window_size=11, sigma=1.5):
    """SSIM for grayscale or RGB image batches in [0, 1].

    fitted, target: (B, C, H, W). RGB SSIM is averaged over channels.
    """
    if fitted.shape[1] == 1:
        return ssim_2d_batch(fitted[:, 0], target[:, 0], window_size, sigma)

    values = [
        ssim_2d_batch(fitted[:, c], target[:, c], window_size, sigma)
        for c in range(fitted.shape[1])
    ]
    return torch.stack(values, dim=1).mean(dim=1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _stats(arr):
    if not arr:
        return None
    a = np.asarray(arr, dtype=np.float64)
    return {
        'mean': float(a.mean()),
        'median': float(np.median(a)),
        'std': float(a.std()),
        'min': float(a.min()),
        'max': float(a.max()),
    }


# ---------------------------------------------------------------------------
# Batched SIREN forward
# ---------------------------------------------------------------------------

def batched_forward(model, phi_batch):
    """Batched forward for ModulatedSIREN / ModulatedSIREN3D.

    The shipped ``ModulatedSIREN.forward`` only accepts a single 1D phi: it
    writes per-layer shifts into each ``SineAffine.shift`` attribute whose
    shape is ``(hidden,)``, i.e. scalar-per-layer, not batch-per-layer.
    This helper lets us push a whole batch of modulators through the net
    in one pass by computing shifts locally and broadcasting them through
    the per-layer affines, never touching the shared ``.shift`` tensors.

    phi_batch: (B, modul_features)
    returns:   (B, N, 1) where N = H*W (2D) or H*W*D (voxel).

    Gradients flow back to phi_batch; backbone params must be frozen for
    eval (main() handles that).
    """
    siren = model.siren
    hidden = siren.hidden_features
    num_layers = siren.num_layers
    B = phi_batch.shape[0]
    shift_all = model.modul(phi_batch)  # (B, hidden * num_layers)
    coord = model.meshgrid  # (N, in_dim); in_dim = 2 or 3
    x = model.fourier(coord) if hasattr(model, 'fourier') else coord
    for i in range(num_layers):
        layer = siren.net[i]
        x_lin = layer.affine(x)
        if i == 0:
            # (N, hidden) -> (B, N, hidden). Shift is per-sample, so we must
            # broadcast across N and the batch dim.
            x_lin = x_lin.unsqueeze(0).expand(B, -1, -1)
        # else x_lin is already (B, N, hidden); nn.Linear acts on last dim.
        shift_i = shift_all[:, i * hidden:(i + 1) * hidden]  # (B, hidden)
        x_lin = x_lin + shift_i.unsqueeze(1)  # (B, N, hidden)
        x = torch.sin(layer.freq * x_lin)
    return siren.hidden2rgb(x)  # (B, N, 1)


# ---------------------------------------------------------------------------
# Per-batch loss / metrics
# ---------------------------------------------------------------------------

def _loss_batch(fitted, target, voxels):
    """Sum-over-batch of per-image mean-squared error, ready for backward.

    fitted: (B, N, 1). target: (B, N, 1) for 2D, (B, N) for voxel.
    Using sum (not mean) over the batch dim means each sample's gradient
    w.r.t. its own modulator equals the gradient it would receive under
    single-image SGD / Adam / LBFGS -- per-image dynamics are preserved.
    """
    f = fitted.squeeze(-1)
    t = target.squeeze(-1) if target.dim() == 3 and target.shape[-1] == 1 else target
    per_image_mse = ((f - t) ** 2).reshape(fitted.shape[0], -1).mean(dim=1)  # (B,)
    return per_image_mse.sum()


def _metrics_batch(fitted, target, image_for_ssim, image_shape, voxels):
    """Per-image (mse_list, psnr_list, ssim_list_or_None), each length B.

    SSIM (2D only) is computed batched via ssim_image_batch. PSNR is derived
    directly from per-image MSE with data_range = 1.0.
    """
    f = fitted.squeeze(-1)
    t = target.squeeze(-1) if target.dim() == 3 and target.shape[-1] == 1 else target
    mse_per = ((f - t) ** 2).reshape(fitted.shape[0], -1).mean(dim=1)  # (B,)
    psnr_per = 10.0 * torch.log10(1.0 / torch.clamp(mse_per, min=1e-12))
    if image_for_ssim is not None:
        B = fitted.shape[0]
        H, W, C = image_shape
        fitted_2d = f.view(B, H, W, C).permute(0, 3, 1, 2).clamp(0.0, 1.0)
        ssim_list = ssim_image_batch(fitted_2d, image_for_ssim).detach().cpu().numpy().tolist()
    else:
        ssim_list = None
    return (mse_per.detach().cpu().numpy().tolist(),
            psnr_per.detach().cpu().numpy().tolist(),
            ssim_list)


# ---------------------------------------------------------------------------
# Per-batch inner-loop fit (one forward+backward per step covers all B samples)
# ---------------------------------------------------------------------------

def _fit_and_snapshot_sgd_batch(model, image_for_loss, image_for_ssim, image_shape,
                                voxels, modul_features, step_set, max_step,
                                inner_lr, device):
    """Batched inner loop. The whole batch's modulator is one tensor of
    shape (B, modul_features) and one forward + one backward per inner
    step covers all B samples. Snapshots are per-image metric lists of
    length B at each requested step."""
    B = image_for_loss.shape[0]
    modulator = torch.zeros(B, modul_features, device=device, requires_grad=True)
    optimizer = (optim.Adam if voxels else optim.SGD)([modulator], lr=inner_lr)
    snapshots = {}
    for step in range(1, max_step + 1):
        optimizer.zero_grad()
        fitted = batched_forward(model, modulator)
        loss = _loss_batch(fitted, image_for_loss, voxels)
        loss.backward()
        optimizer.step()
        if step in step_set:
            with torch.no_grad():
                fitted_post = batched_forward(model, modulator)
                snapshots[step] = _metrics_batch(
                    fitted_post, image_for_loss, image_for_ssim, image_shape, voxels,
                )
    return snapshots


def _fit_and_snapshot_lbfgs_batch(model, image_for_loss, image_for_ssim, image_shape,
                                  voxels, modul_features, checkpoints, inner_lr, device):
    """Batched LBFGS: one independent run per checkpoint, each over the joint
    (B, modul_features) modulator.

    Caveat: L-BFGS's rank-m inverse-Hessian approximation couples samples
    across the batch dim via its quasi-Newton updates even though the true
    Hessian is block-diagonal (loss_i depends only on modulator_i). For
    strict per-image parity pass ``--batch-size 1``. In practice the
    batched variant converges close enough for MNIST benchmarks.
    """
    B = image_for_loss.shape[0]
    snapshots = {}
    for target_iters in checkpoints:
        modulator = torch.zeros(B, modul_features, device=device, requires_grad=True)
        optimizer = optim.LBFGS(
            [modulator], lr=inner_lr, max_iter=target_iters,
            line_search_fn='strong_wolfe',
        )

        def closure():
            optimizer.zero_grad()
            fitted = batched_forward(model, modulator)
            loss = _loss_batch(fitted, image_for_loss, voxels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            fitted = batched_forward(model, modulator)
            snapshots[target_iters] = _metrics_batch(
                fitted, image_for_loss, image_for_ssim, image_shape, voxels,
            )
    return snapshots


def _prep_image_batch(image_batch, voxels, device):
    """Move batch to device and return the shapes the fit loops expect.

    Returns:
      image_for_loss: (B, HW, C) for 2D, (B, HWD) for voxel.
      image_for_ssim: (B, C, H, W) for 2D, None for voxel.
    """
    image_batch = image_batch.to(device, non_blocking=True)
    B = image_batch.shape[0]
    if voxels:
        return image_batch.reshape(B, -1), None
    image_for_ssim = image_batch
    image_for_loss = image_batch.permute(0, 2, 3, 1).reshape(B, -1, image_batch.shape[1])
    return image_for_loss, image_for_ssim


# ---------------------------------------------------------------------------
# Split-level evaluation
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, iter_checkpoints, inner_lr, lbfgs, voxels, device,
                   image_shape, max_samples=None):
    modul_features = model.modul_features
    step_set = set(iter_checkpoints)
    max_step = max(iter_checkpoints)
    accum = {s: {'mse': [], 'psnr': [], 'ssim': []} for s in iter_checkpoints}

    dataset_size = len(loader.dataset)
    target_samples = dataset_size if max_samples is None else min(max_samples, dataset_size)
    samples_done = 0
    prog = tqdm(loader, total=len(loader))
    last_key = iter_checkpoints[-1] if lbfgs else max_step
    for image_batch, _ in prog:
        if samples_done >= target_samples:
            break
        B = image_batch.shape[0]
        if samples_done + B > target_samples:
            B = target_samples - samples_done
            image_batch = image_batch[:B]

        image_for_loss, image_for_ssim = _prep_image_batch(image_batch, voxels, device)

        if lbfgs:
            snapshots = _fit_and_snapshot_lbfgs_batch(
                model, image_for_loss, image_for_ssim, image_shape, voxels,
                modul_features, iter_checkpoints, inner_lr, device,
            )
        else:
            snapshots = _fit_and_snapshot_sgd_batch(
                model, image_for_loss, image_for_ssim, image_shape, voxels,
                modul_features, step_set, max_step, inner_lr, device,
            )

        for s, (mse_list, psnr_list, ssim_list) in snapshots.items():
            accum[s]['mse'].extend(mse_list)
            accum[s]['psnr'].extend(psnr_list)
            if ssim_list is not None:
                accum[s]['ssim'].extend(ssim_list)

        samples_done += B
        last_psnr = accum[last_key]['psnr']
        if last_psnr:
            prog.set_description(
                f'samples={samples_done}/{target_samples} PSNR@final mean {np.mean(last_psnr):.2f}'
            )

    out = {}
    for s in iter_checkpoints:
        row = {'mse': _stats(accum[s]['mse']), 'psnr': _stats(accum[s]['psnr'])}
        ssim_stats = _stats(accum[s]['ssim'])
        if ssim_stats is not None:
            row['ssim'] = ssim_stats
        out[str(s)] = row
    return out, samples_done


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description='Reconstruction evaluation for trained SIREN backbones '
                    '(MSE / PSNR / SSIM at multiple inner-loop iteration counts).',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained SIREN .pth file.')
    parser.add_argument('--dataset', choices=['mnist', 'fmnist', 'cifar10', 'modelnet'], required=True)
    parser.add_argument('--data-path', type=str, default='..')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Overridden by checkpoint.model_args if present.')
    parser.add_argument('--mod-dim', type=int, default=512,
                        help='Overridden by checkpoint.model_args if present.')
    parser.add_argument('--depth', type=int, default=10,
                        help='Overridden by checkpoint.model_args if present.')
    parser.add_argument('--inr-type', choices=['siren', 'fourier_siren'], default='siren',
                        help='Coordinate INR backbone type. Overridden by checkpoint.model_args if present.')
    parser.add_argument('--fourier-num-freqs', type=int, default=64,
                        help='Number of random Fourier frequencies for --inr-type fourier_siren.')
    parser.add_argument('--fourier-sigma', type=float, default=10.0,
                        help='Stddev of Gaussian Fourier frequency matrix B.')
    parser.add_argument('--fourier-include-input', action='store_true', default=False,
                        help='Concatenate raw (x,y) coordinates to Fourier features.')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', choices=['train', 'test', 'both'], default='both')
    parser.add_argument('--iter-checkpoints', type=str, default='5,20,50,100,200',
                        help='Comma-separated list of inner-loop step counts at which '
                             'to snapshot reconstruction metrics.')
    parser.add_argument('--inner-lr', type=float, default=0.01)
    parser.add_argument('--lbfgs', action='store_true', default=False,
                        help='Use LBFGS instead of SGD/Adam. NOTE: one independent '
                             'LBFGS run per iter-checkpoint since history cannot be '
                             'shared across different max_iter settings.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Inner-loop batch size: B modulators are fit in parallel '
                             'per forward/backward pass. Gradient w.r.t. each sample '
                             "'s modulator equals the single-image gradient (loss is "
                             'summed, not averaged). Raise for faster throughput, '
                             'lower for memory-constrained GPUs. Pass 1 for strict '
                             'LBFGS per-image parity.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Per-split cap on the number of samples evaluated. '
                             'Default: full split (slow on MNIST train).')
    parser.add_argument('--output', type=str, default=None,
                        help='Destination JSON. Default: '
                             '<checkpoint_dir>/reconstruction_eval.json.')
    parser.add_argument('--variant', choices=variants.available(), default='vanilla',
                        help='SIREN variant used at training time. Must match the '
                             'variant recorded in the checkpoint so architecture '
                             'wrappers line up before load_state_dict.')
    variants.add_all_variant_args(parser)
    return parser.parse_args()


def _build_model(args, ckpt_model_args, device):
    model_args = {
        'hidden_dim': args.hidden_dim,
        'mod_dim': args.mod_dim,
        'depth': args.depth,
        'height': 32 if args.dataset == 'cifar10' else 28,
        'width': 32 if args.dataset == 'cifar10' else 28,
        'out_features': 3 if args.dataset == 'cifar10' else 1,
        'inr_type': args.inr_type,
        'fourier_num_freqs': args.fourier_num_freqs,
        'fourier_sigma': args.fourier_sigma,
        'fourier_include_input': args.fourier_include_input,
    }
    if ckpt_model_args:
        for k in ('hidden_dim', 'mod_dim', 'depth', 'height', 'width', 'out_features',
                  'inr_type', 'fourier_num_freqs', 'fourier_sigma', 'fourier_include_input'):
            if k in ckpt_model_args and ckpt_model_args[k] != model_args[k]:
                print(f"[eval] override --{k.replace('_','-')} "
                      f"{model_args[k]} -> {ckpt_model_args[k]} (from checkpoint.model_args)")
                model_args[k] = ckpt_model_args[k]
    else:
        print("[eval] checkpoint has no 'model_args'; using CLI --hidden-dim / --mod-dim / --depth.")

    if args.dataset == 'modelnet':
        if model_args.get('inr_type', 'siren') != 'siren':
            raise SystemExit("fourier_siren is currently supported for 2D image datasets only.")
        H, W, D = 15, 15, 15
        model = ModulatedSIREN3D(
            height=H, width=W, depth=D,
            hidden_features=model_args['hidden_dim'],
            num_layers=model_args['depth'],
            modul_features=model_args['mod_dim'],
        )
        image_shape = (H, W, D)
    else:
        if model_args.get('inr_type', 'siren') == 'fourier_siren':
            model = ModulatedFourierSIREN(
                height=model_args['height'], width=model_args['width'],
                hidden_features=model_args['hidden_dim'],
                num_layers=model_args['depth'],
                modul_features=model_args['mod_dim'],
                device=device,
                out_features=model_args['out_features'],
                fourier_num_freqs=model_args.get('fourier_num_freqs', 64),
                fourier_sigma=model_args.get('fourier_sigma', 10.0),
                fourier_include_input=model_args.get('fourier_include_input', False),
            )
        else:
            model = ModulatedSIREN(
                height=model_args['height'], width=model_args['width'],
                hidden_features=model_args['hidden_dim'],
                num_layers=model_args['depth'],
                modul_features=model_args['mod_dim'],
                device=device,
                out_features=model_args['out_features'],
            )
        image_shape = (model_args['height'], model_args['width'], model_args['out_features'])
    return model, model_args, image_shape


def _get_loaders(args):
    bs = args.batch_size
    loaders = {}
    if args.dataset == 'modelnet':
        if args.split in ('train', 'both'):
            loaders['train'] = get_modelnet_loader(
                train=True, batch_size=bs, resample_shape=(15, 15, 15),
            )
        if args.split in ('test', 'both'):
            loaders['test'] = get_modelnet_loader(
                train=False, batch_size=bs, resample_shape=(15, 15, 15),
            )
    else:
        if args.dataset == 'cifar10':
            if args.split in ('train', 'both'):
                loaders['train'] = get_cifar10_loader(
                    args.data_path, train=True, batch_size=bs,
                )
            if args.split in ('test', 'both'):
                loaders['test'] = get_cifar10_loader(
                    args.data_path, train=False, batch_size=bs,
                )
        else:
            fashion = (args.dataset == 'fmnist')
            if args.split in ('train', 'both'):
                loaders['train'] = get_mnist_loader(
                    args.data_path, train=True, batch_size=bs, fashion=fashion,
                )
            if args.split in ('test', 'both'):
                loaders['test'] = get_mnist_loader(
                    args.data_path, train=False, batch_size=bs, fashion=fashion,
                )
    return loaders


def _print_table(split_name, n, checkpoints, at_iters):
    print(f"\n=== Reconstruction eval: {split_name} (n={n}) ===")
    has_ssim = any('ssim' in at_iters[str(s)] for s in checkpoints)
    if has_ssim:
        print(f"{'iters':>6}  {'MSE':>10}  {'PSNR mean/med':>14}  {'SSIM mean':>10}")
    else:
        print(f"{'iters':>6}  {'MSE':>10}  {'PSNR mean/med':>14}")
    for s in checkpoints:
        r = at_iters[str(s)]
        mse_m = r['mse']['mean']
        p_m = r['psnr']['mean']
        p_med = r['psnr']['median']
        if has_ssim and 'ssim' in r:
            s_m = r['ssim']['mean']
            print(f"{s:>6}  {mse_m:>10.3e}  {p_m:>6.2f} / {p_med:<5.2f}  {s_m:>10.4f}")
        else:
            print(f"{s:>6}  {mse_m:>10.3e}  {p_m:>6.2f} / {p_med:<5.2f}")


def main():
    args = get_args()
    set_random_seeds(args.seed, args.device)
    device = args.device

    iter_checkpoints = sorted({int(x) for x in args.iter_checkpoints.split(',') if x.strip()})
    if not iter_checkpoints:
        raise SystemExit("--iter-checkpoints must be a non-empty comma-separated list of ints")
    print(f"[eval] iter_checkpoints = {iter_checkpoints}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"[eval] loaded checkpoint: {args.checkpoint}")
    if 'variant' in ckpt and ckpt['variant'] != args.variant:
        print(f"[eval] WARNING: checkpoint variant '{ckpt['variant']}' differs from "
              f"--variant '{args.variant}'. Continuing with '{args.variant}'.")

    model, model_args, image_shape = _build_model(args, ckpt.get('model_args'), device)
    model = model.to(device)
    model = variants.build(args.variant, model, args)
    model.load_state_dict(ckpt['state_dict'])
    # Freeze backbone; only the inner-loop modulator is optimized.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    loaders = _get_loaders(args)
    voxels = (args.dataset == 'modelnet')

    results_per_split = {}
    for split_name, loader in loaders.items():
        print(f"\n[eval] split='{split_name}' n={len(loader)} (cap={args.max_samples})")
        at_iters, n_used = evaluate_split(
            model, loader, iter_checkpoints, args.inner_lr, args.lbfgs, voxels, device,
            image_shape, max_samples=args.max_samples,
        )
        results_per_split[split_name] = {'n_samples': n_used, 'at_iters': at_iters}

    output_record = {
        'checkpoint': osp.abspath(args.checkpoint),
        'variant': args.variant,
        'variant_args': variants._extract_variant_args(args, args.variant),
        'model_args': dict(model_args, dataset=args.dataset),
        'inner_lr': args.inner_lr,
        'lbfgs': args.lbfgs,
        'batch_size': args.batch_size,
        'iter_checkpoints': iter_checkpoints,
        'max_samples': args.max_samples,
        'split': args.split,
        'results': results_per_split,
    }

    output_path = args.output or osp.join(
        osp.dirname(args.checkpoint) or '.', 'reconstruction_eval.json',
    )
    os.makedirs(osp.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as fh:
        json.dump(output_record, fh, indent=2)
    print(f"\n[eval] wrote JSON summary to {output_path}")

    for split_name, res in results_per_split.items():
        _print_table(split_name, res['n_samples'], iter_checkpoints, res['at_iters'])


if __name__ == '__main__':
    main()
