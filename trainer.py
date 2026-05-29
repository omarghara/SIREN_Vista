import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_cifar10_loader, get_mnist_loader
from dataloader_modelnet import get_modelnet_loader
from SIREN import (
    ModulatedFourierSIREN,
    ModulatedSIREN,
    ModulatedSIREN3D,
    ModulatedFINER,
    ModulatedFourierLSA,
    SpatialModulatedINR,
)
from utils import adjust_learning_rate
from tqdm import tqdm
import os
import argparse
from utils import set_random_seeds
import variants
from diagnostics import layer_sigmas, format_sigmas_one_liner


def write_checkpoint_summary(savedir, ckpt):
    """Write human-readable checkpoint_summary.md next to modSiren.pth."""
    run_slug = ckpt.get('model_name') or os.path.basename(savedir.rstrip(os.sep))
    rel_ckpt = f"{savedir}/modSiren.pth"
    lines = [
        "# Backbone checkpoint summary",
        "",
        f"- **path**: `{rel_ckpt}`",
        f"- **model_name**: `{run_slug}`",
        f"- **variant**: `{ckpt.get('variant', 'vanilla')}`",
        f"- **epoch (best)**: {ckpt.get('epoch')}",
        f"- **loss (best mean outer loss)**: {ckpt.get('loss')}",
    ]
    if ckpt.get('num_epochs_requested') is not None:
        lines.append(f"- **num_epochs requested**: {ckpt['num_epochs_requested']}")
    ers = ckpt.get('epoch_range_start')
    ere = ckpt.get('epoch_range_end')
    if ers is not None and ere is not None:
        lines.append(f"- **training epoch range**: {ers}–{ere}")
    lines.append("")

    for section, key in (("model_args", "model_args"), ("variant_args", "variant_args")):
        args_dict = ckpt.get(key)
        if not args_dict:
            continue
        lines.append(f"## {section}")
        lines.append("")
        for k in sorted(args_dict.keys()):
            lines.append(f"- `{k}`: {repr(args_dict[k])}")
        lines.append("")

    summary_path = os.path.join(savedir, "checkpoint_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _prep_2d_batch(images, device):
    """Return images as (B, H*W, C) to match ModulatedSIREN output."""
    return images.to(device).permute(0, 2, 3, 1).reshape(images.size(0), -1, images.size(1))


def _build_2d_model(args, height, width, out_features):
    if getattr(args, 'spatial_modulation', False):
        model = SpatialModulatedINR(
            height=height,
            width=width,
            hidden_features=args.hidden_dim,
            num_layers=args.depth,
            latent_spatial_dim=args.latent_spatial_dim,
            latent_dim=args.latent_dim,
            base_inr_type=args.inr_type,
            spatial_interp=args.spatial_interp,
            use_local_coords=args.use_local_coords,
            modulation_type=args.modulation_type,
            freq=args.freq,
            device=args.device,
            out_features=out_features,
            fourier_num_freqs=args.fourier_num_freqs,
            fourier_sigma=args.fourier_sigma,
            fourier_include_input=args.fourier_include_input,
            first_bias_scale=args.finer_first_bias_scale,
            scale_req_grad=args.finer_scale_req_grad,
            lsa_num_harmonics=args.lsa_num_harmonics,
            lsa_init_scale=args.lsa_init_scale,
            lsa_include_linear=not args.lsa_no_linear,
        )
        print(f"[build] {type(model).__name__}  base_inr_type={model.base_inr_type}  "
              f"is_spatial={model.is_spatial}  phi_shape={tuple(model.phi_shape)}  "
              f"phi_numel={int(model.phi_numel)}")
        return model

    if args.inr_type == 'fourier_siren':
        model = ModulatedFourierSIREN(
            height=height,
            width=width,
            hidden_features=args.hidden_dim,
            num_layers=args.depth,
            modul_features=args.mod_dim,
            device=args.device,
            out_features=out_features,
            freq=args.freq,
            fourier_num_freqs=args.fourier_num_freqs,
            fourier_sigma=args.fourier_sigma,
            fourier_include_input=args.fourier_include_input,
        )
    elif args.inr_type == 'fourier_lsa':
        model = ModulatedFourierLSA(
            height=height,
            width=width,
            hidden_features=args.hidden_dim,
            num_layers=args.depth,
            modul_features=args.mod_dim,
            device=args.device,
            out_features=out_features,
            fourier_num_freqs=args.fourier_num_freqs,
            fourier_sigma=args.fourier_sigma,
            fourier_include_input=args.fourier_include_input,
            lsa_num_harmonics=args.lsa_num_harmonics,
            lsa_init_scale=args.lsa_init_scale,
            lsa_include_linear=not args.lsa_no_linear,
        )
    elif args.inr_type == 'finer':
        model = ModulatedFINER(
            height=height,
            width=width,
            hidden_features=args.hidden_dim,
            num_layers=args.depth,
            modul_features=args.mod_dim,
            device=args.device,
            out_features=out_features,
            freq=args.freq,
            first_bias_scale=args.finer_first_bias_scale,
            scale_req_grad=args.finer_scale_req_grad,
        )
    elif args.inr_type == 'siren':
        model = ModulatedSIREN(
            height=height,
            width=width,
            hidden_features=args.hidden_dim,
            num_layers=args.depth,
            modul_features=args.mod_dim,
            device=args.device,
            out_features=out_features,
            freq=args.freq,
        )
    else:
        raise ValueError(f"Unknown --inr-type {args.inr_type!r}")

    print(f"[build] {type(model).__name__}  is_spatial={model.is_spatial}  "
          f"phi_shape={tuple(model.phi_shape)}  phi_numel={int(model.phi_numel)}")
    return model


def fit(
        model,
        data_loader,
        outer_optimizer,
        outer_criterion,
        epoch_id,
        inner_steps=3,
        inner_optim='sgd',
        inner_lr=0.01,
        voxels=False,
        penalty_fn=None,
        log_sigmas_every=0,
):
    """
    Fit the INR for each specific sample for inner_steps steps to perform meta-learning.
    :param model: Meta-network INR.
    :param data_loader: Dataloader for dataset to train on.
    :param outer_optimizer: Meta-learning optimizer.
    :param outer_criterion: Meta-learning training objective.
    :param epoch_id: Epoch number.
    :param inner_steps: Number of internal, per-sample optimization steps for INR optimization.
    :param inner_optim: Optimizer for internal, per-sample optimization.
    :param inner_lr: Learn-rate for internal, per-sample optimization.
    :param voxels: whether to use 3d data (.e.g modelnet) or 2d
    :param penalty_fn: Optional callable (model -> scalar Tensor) whose value is
        added to the outer loss before backprop. Used by SIREN variants to
        inject training-time regularizers (e.g. soft-Lipschitz penalty).
    :param log_sigmas_every: If > 0, emit a one-line per-layer spectral-norm
        report every N outer batches via ``tqdm.write`` (so the progress bar
        is preserved). Useful for calibrating soft-Lipschitz L / lambda.
        Default: 0 (off).
    :return: Average representation loss.
    """
  
    losses = []
    mse_losses = []
    pen_losses = []
    device = next(iter(model.parameters())).device
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))
    for batch_idx, (images, labels) in enumerate(prog_bar):
        batch_size = images.size(0)
        images = images.squeeze().to(device) if voxels else _prep_2d_batch(images, device)
        modulators = []
        # Inner loop.
        for batch_id in range(batch_size):
            # model.init_phi() returns a zero tensor matching this model's phi_shape:
            # (modul_features,) for global Functa, (s, s, c) for Spatial Functa.
            modulator = model.init_phi(device=device).float()
            modulator.requires_grad = True
            if voxels:
                inner_optimizer = optim.Adam([modulator], lr=inner_lr)
            else:
                if inner_optim == 'adam':
                    inner_optimizer = optim.Adam([modulator], lr=inner_lr)
                else:
                    inner_optimizer = optim.SGD([modulator], lr=inner_lr)
            
            # Inner Optimization.
            for step in range(inner_steps):
                # Inner optimizer step.
                inner_optimizer.zero_grad()
                fitted = model(modulator)
           
                inner_loss = inner_criterion(fitted.T, images[batch_id].flatten()[None]) if voxels else inner_criterion(fitted, images[batch_id])
                inner_loss.backward()
             
                # Update.
                inner_optimizer.step()
            modulator.requires_grad = False
            modulators.append(modulator)

        outer_optimizer.zero_grad()
        outer_loss = torch.tensor(0).to(device).float()
        for batch_id in range(batch_size):
            modulator = modulators[batch_id]
            # Outer Optimization.
            fitted = model(modulator)
            outer_loss += (outer_criterion(fitted.T, images[batch_id].flatten()[None]) if voxels else outer_criterion(fitted, images[batch_id])) / batch_size

        mse_component = outer_loss.detach().item()
        pen_component = 0.0
        if penalty_fn is not None:
            pen = penalty_fn(model)
            pen_component = pen.detach().item()
            outer_loss = outer_loss + pen

        # Outer optimizer step.
        outer_loss.backward()
        # Clip the gradient.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        outer_optimizer.step()
        losses.append(outer_loss.item())
        mse_losses.append(mse_component)
        pen_losses.append(pen_component)

        prog_bar.set_description(
            f'Epoch {epoch_id} | total {outer_loss.item():.4f} '
            f'| MSE {mse_component:.4f} | pen {pen_component:.4f}'
        )

        if log_sigmas_every > 0 and (batch_idx % log_sigmas_every == 0):
            sigmas = layer_sigmas(model)
            tqdm.write(f'  [sigmas @ epoch {epoch_id} batch {batch_idx}] '
                       f'{format_sigmas_one_liner(sigmas)}')

    n = len(losses)
    avg_total = sum(losses) / n
    avg_mse = sum(mse_losses) / n
    avg_pen = sum(pen_losses) / n
    print(f'epoch: {epoch_id}, total: {avg_total:.6f}, '
          f'MSE: {avg_mse:.6f}, pen: {avg_pen:.6f}')
    return avg_total


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--ext-lr', type=float, default=5e-6, help='external optimization loop lr')
    parser.add_argument('--int-lr', type=float, default=0.01, help='internal optimization loop lr')
    parser.add_argument('--batch-size', type=int, default=128, help='optimization minibatch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--depth', type=int, default=10, help='SIREN depth')
    parser.add_argument('--inr-type', choices=['siren', 'fourier_siren', 'finer', 'fourier_lsa'], default='siren',
                        help='Coordinate INR backbone type.')
    parser.add_argument('--lsa-num-harmonics', type=int, default=8,
                    help='Number of harmonics K in learnable spectral activation.')

    parser.add_argument('--lsa-init-scale', type=float, default=1e-3,
                        help='Initial std scale for learnable spectral activation coefficients.')

    parser.add_argument('--lsa-no-linear', action='store_true', default=False,
                        help='If set, remove the identity term u from LSA activation.')
    parser.add_argument('--fourier-num-freqs', type=int, default=64,
                        help='Number of random Fourier frequencies for --inr-type fourier_siren.')
    parser.add_argument('--fourier-sigma', type=float, default=10.0,
                        help='Stddev of Gaussian Fourier frequency matrix B.')
    parser.add_argument('--fourier-include-input', action='store_true', default=False,
                        help='Concatenate raw (x,y) coordinates to Fourier features.')
    parser.add_argument('--freq', type=float, default=30.0,
                        help='ω0 (angular frequency) used by SIREN, FINER, and Fourier-SIREN '
                             'hidden layers: sin(ω0(Wx + b + shift)). Default 30.')
    parser.add_argument('--dataset', choices=["mnist", "fmnist", "cifar10", "modelnet"], help="Train for MNIST, Fashion-MNIST, CIFAR-10, or ModelNet10")
    parser.add_argument('--finer-first-bias-scale', type=float, default=1.0,
                        help='FINER first-layer bias init range: U(-scale, scale). '
                            'This controls the supported frequency set.')
    parser.add_argument('--finer-scale-req-grad', action='store_true', default=False,
                        help='If set, allow gradients through FINER scale = |z| + 1. '
                            'Default false is closer to the reference implementation.')
    parser.add_argument('--num-epochs', type=int, default=6, help='number of epochs for external optimization')
    parser.add_argument('--data-path', type=str, default='..', help='path to MNIST, FMNIST or ModelNet10 dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a .pth checkpoint to resume training from. '
                             'Loads model weights, optimizer state (if present), '
                             'epoch counter, and best_loss.')
    parser.add_argument('--variant', choices=variants.available(), default='vanilla',
                        help='SIREN variant to train.')
    parser.add_argument('--model-name', '--run-name', dest='model_name',
                        type=str, default=None,
                        help='Optional subdirectory name override under '
                             'model_{dataset}/. Defaults to the variant slug. '
                             '--run-name is kept as a backwards-compatible alias.')
    parser.add_argument('--log-sigmas-every', type=int, default=0,
                        help='If > 0, print per-layer spectral norms every N '
                             'outer batches. Useful for calibrating soft-Lipschitz '
                             'L / lambda. Default: 0 (off).')

    parser.add_argument('--inner-steps', type=int, default=3,
                    help='Number of inner-loop phi adaptation steps during meta-training.')
    parser.add_argument('--inner-optim', choices=['sgd', 'adam'], default='sgd',
                    help='Optimizer for inner-loop phi adaptation during meta-training.')

    # Spatial Functa flags. All default off => existing global Functa behavior unchanged.
    parser.add_argument('--spatial-modulation', action='store_true', default=False,
                        help='Use a spatial latent grid (Spatial Functa) instead of a '
                             'single global modulation vector. When set, --inr-type '
                             'selects the underlying backbone.')
    parser.add_argument('--latent-spatial-dim', type=int, default=8,
                        help='Spatial side s of the latent grid (phi has shape (s, s, c)). '
                             'Used only when --spatial-modulation is set.')
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Per-cell latent channels c of the latent grid (phi has '
                             'shape (s, s, c)). Used only when --spatial-modulation is set.')
    parser.add_argument('--spatial-interp', choices=['nearest'], default='nearest',
                        help='Spatial latent interpolation. Only nearest (1-NN) is supported.')
    parser.add_argument('--use-local-coords', action='store_true', default=False,
                        help='If set, feed each pixel its local coordinate (coord*s - cell) '
                             'in [0, 1] instead of the global coordinate. Paper default.')
    parser.add_argument('--modulation-type', choices=['shift'], default='shift',
                        help='Spatial modulation type. Only shift is supported.')

    variants.add_all_variant_args(parser)
    return parser.parse_args()

if __name__ == '__main__':
    # Training Parameters.
  
    args = get_args()
    
    device = args.device
    set_random_seeds(args.seed,device)
    if args.dataset == "modelnet":
        if args.inr_type != 'siren':
            raise SystemExit("--inr-type fourier_siren is currently supported for 2D image datasets only.")
        resample_shape = (15,15,15) #we use this resampling in all experiments
        dataloader = get_modelnet_loader(train=True, batch_size=args.batch_size, resample_shape=resample_shape)
        modSiren = ModulatedSIREN3D(height=resample_shape[0], width=resample_shape[1], depth=resample_shape[2],\
            hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #we use a mod dim of 2048 in our exps
  
    else:
        if args.dataset == "cifar10":
            dataloader = get_cifar10_loader(args.data_path, train=True, batch_size=args.batch_size)
            modSiren = _build_2d_model(args, height=32, width=32, out_features=3)
        else:
            dataloader = get_mnist_loader(args.data_path, train=True, batch_size=args.batch_size, fashion = args.dataset=="fmnist")
            modSiren = _build_2d_model(args, height=28, width=28, out_features=1) #28,28 is mnist and fmnist dims

        
      
    
    modSiren = modSiren.to(args.device)
    modSiren = variants.build(args.variant, modSiren, args)
    optimizer = optim.Adam(modSiren.parameters(), lr=args.ext_lr)
    criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    penalty_fn = lambda m: variants.penalty(args.variant, m, args)

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

    if args.model_name is not None:
        run_slug = args.model_name
    else:
        run_slug = variants.slug(args.variant, args)
    savedir = f"model_{args.dataset}/{run_slug}" if run_slug else f"model_{args.dataset}"

    os.makedirs(savedir, exist_ok=True)
    initial_start_epoch = start_epoch
    epoch_range_end = initial_start_epoch + args.num_epochs - 1
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        loss = fit(
            modSiren, dataloader, optimizer, criterion, epoch,
            inner_steps=args.inner_steps,
            inner_optim=args.inner_optim,
            inner_lr=args.int_lr,
            voxels=args.dataset=='modelnet',
            penalty_fn=penalty_fn,
            log_sigmas_every=args.log_sigmas_every,
        )
        if loss < best_loss:
            best_loss = loss
            ckpt_data = {
                'epoch': epoch,
                'state_dict': modSiren.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'variant': args.variant,
                'model_name': run_slug,
                'num_epochs_requested': args.num_epochs,
                'epoch_range_start': initial_start_epoch,
                'epoch_range_end': epoch_range_end,
                'variant_args': variants._extract_variant_args(args, args.variant),
                'model_args': {
                    'dataset': args.dataset,
                    'hidden_dim': args.hidden_dim,
                    'mod_dim': args.mod_dim,
                    'depth': args.depth,
                    'height': getattr(modSiren, 'height', None),
                    'width': getattr(modSiren, 'width', None),
                    'out_features': getattr(modSiren, 'out_features', 1),
                    'inr_type': args.inr_type,
                    'fourier_num_freqs': args.fourier_num_freqs,
                    'fourier_sigma': args.fourier_sigma,
                    'fourier_include_input': args.fourier_include_input,
                    'freq': args.freq,
                    'finer_first_bias_scale': args.finer_first_bias_scale,
                    'finer_scale_req_grad': args.finer_scale_req_grad,
                    'coord_normalization': 'zero_one_pixel_centers',
                    'inner_optim': args.inner_optim,
                    'inner_lr': args.int_lr,
                    'inner_steps': args.inner_steps,
                    'lsa_num_harmonics': args.lsa_num_harmonics,
                    'lsa_init_scale': args.lsa_init_scale,
                    'lsa_include_linear': not args.lsa_no_linear,
                    'spatial_modulation': bool(getattr(args, 'spatial_modulation', False)),
                    'latent_spatial_dim': args.latent_spatial_dim,
                    'latent_dim': args.latent_dim,
                    'spatial_interp': args.spatial_interp,
                    'use_local_coords': bool(args.use_local_coords),
                    'modulation_type': args.modulation_type,
                    'is_spatial': bool(getattr(modSiren, 'is_spatial', False)),
                    'phi_shape': tuple(getattr(modSiren, 'phi_shape', (args.mod_dim,))),
                    'phi_numel': int(getattr(modSiren, 'phi_numel', args.mod_dim)),
                },
            }
            ckpt_path = f'{savedir}/modSiren.pth'
            torch.save(ckpt_data, ckpt_path)
            write_checkpoint_summary(savedir, ckpt_data)

