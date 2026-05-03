import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_cifar10_loader, get_mnist_loader
from dataloader_modelnet import get_modelnet_loader
from SIREN import ModulatedFourierSIREN, ModulatedSIREN, ModulatedSIREN3D
from utils import adjust_learning_rate
from tqdm import tqdm
import os
import argparse
from utils import set_random_seeds
import variants
from diagnostics import layer_sigmas, format_sigmas_one_liner


def _prep_2d_batch(images, device):
    """Return images as (B, H*W, C) to match ModulatedSIREN output."""
    return images.to(device).permute(0, 2, 3, 1).reshape(images.size(0), -1, images.size(1))


def _build_2d_model(args, height, width, out_features):
    if args.inr_type == 'fourier_siren':
        return ModulatedFourierSIREN(
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
        )
    if args.inr_type != 'siren':
        raise ValueError(f"Unknown --inr-type {args.inr_type!r}")
    return ModulatedSIREN(
        height=height,
        width=width,
        hidden_features=args.hidden_dim,
        num_layers=args.depth,
        modul_features=args.mod_dim,
        device=args.device,
        out_features=out_features,
    )


def fit(
        model,
        data_loader,
        outer_optimizer,
        outer_criterion,
        epoch_id,
        inner_steps=3,
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
    modul_features = model.modul_features
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))
    for batch_idx, (images, labels) in enumerate(prog_bar):
        batch_size = images.size(0)
        images = images.squeeze().to(device) if voxels else _prep_2d_batch(images, device)
        modulators = []
        # Inner loop.
        for batch_id in range(batch_size):
            modulator = torch.zeros(modul_features).float().to(device)
            modulator.requires_grad=True
            inner_optimizer = (optim.Adam if voxels else optim.SGD)([modulator], lr=inner_lr)
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
    parser.add_argument('--inr-type', choices=['siren', 'fourier_siren'], default='siren',
                        help='Coordinate INR backbone type.')
    parser.add_argument('--fourier-num-freqs', type=int, default=64,
                        help='Number of random Fourier frequencies for --inr-type fourier_siren.')
    parser.add_argument('--fourier-sigma', type=float, default=10.0,
                        help='Stddev of Gaussian Fourier frequency matrix B.')
    parser.add_argument('--fourier-include-input', action='store_true', default=False,
                        help='Concatenate raw (x,y) coordinates to Fourier features.')
    parser.add_argument('--dataset', choices=["mnist", "fmnist", "cifar10", "modelnet"], help="Train for MNIST, Fashion-MNIST, CIFAR-10, or ModelNet10")
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
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        loss = fit(
            modSiren, dataloader, optimizer, criterion, epoch, inner_steps=3,inner_lr=args.int_lr, voxels=args.dataset=='modelnet',
            penalty_fn=penalty_fn,
            log_sigmas_every=args.log_sigmas_every,
        )
        if loss < best_loss:
            best_loss = loss
            torch.save({'epoch': epoch,
                        'state_dict': modSiren.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'variant': args.variant,
                        'model_name': run_slug,
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
                        },
                        }, f'{savedir}/modSiren.pth')

