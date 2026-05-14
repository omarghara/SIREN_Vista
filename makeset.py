import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_cifar10_loader, get_mnist_loader
from torch.utils.data import DataLoader, Subset
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
import joblib
from tqdm import tqdm
import variants


def _prep_2d_image(image, device):
    """Return one image as (H*W, C) to match ModulatedSIREN output."""
    return image.permute(1, 2, 0).reshape(-1, image.size(0)).to(device)


# Create a functaset on MNIST, Fashion MNIST or ModelNet10
def create_functaset(
        model,
        data_loader,
        inner_steps=100,
        inner_lr=0.01,
        voxels=False,
        lbfgs=False,
        inner_optim='sgd',
):
    """
    :param model: INR model for which modulations are fitted.
    :param data_loader: Torch dataloader object for fitting data.
    :param inner_steps: Number of modulation optimization steps.
    :param inner_lr: Learn-rate for modulation optimization.
    :param voxels: Whether we optimize for 3D data or not.
    :param lbfgs: Whether to use L-BFGS or SGD/Adam.
    """
    assert data_loader.batch_size == 1
    functaset = []
    device = 'cuda'
    model = model.cuda()
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))
    _printed_phi_shape = False
    is_spatial = bool(getattr(model, 'is_spatial', False))

    for image, label in prog_bar:
        
        image = image.squeeze().to(device) if voxels else _prep_2d_image(image[0], device)
        # init_phi returns a zero tensor of the model's native phi shape:
        # (modul_features,) for global Functa, (s, s, c) for Spatial Functa.
        modulator = model.init_phi(device=device).float()
        modulator.requires_grad = True

        if not _printed_phi_shape:
            print("[makeset] first phi shape:", tuple(modulator.shape), "is_spatial:", is_spatial)
            _printed_phi_shape = True
        
        def closure():
            inner_optimizer.zero_grad()
            fitted = model(modulator)
            inner_loss = inner_criterion(fitted, image)
            inner_loss.backward()
            return inner_loss

        if lbfgs:
            if voxels:
                image = image.view(1, -1).T
            inner_optimizer = optim.LBFGS([modulator], lr=inner_lr, max_iter=inner_steps, line_search_fn="strong_wolfe")
            inner_optimizer.step(closure)
            with torch.no_grad():
                mse = torch.nn.MSELoss()(model(modulator),image).item()
        else:
            if voxels:
                opt_cls = optim.Adam
            else:
                opt_cls = optim.Adam if inner_optim == 'adam' else optim.SGD

            inner_optimizer = opt_cls([modulator], lr=inner_lr)
            
            mse = 0
            # Inner Optimization.
            for step in range(inner_steps):
                fitted = model(modulator)
                inner_loss = inner_criterion(fitted.T, image.flatten()[None]) if voxels else inner_criterion(fitted, image)
                mse = inner_loss.item()

                # Inner optimizer step.
                inner_optimizer.zero_grad()
                inner_loss.backward()
                
                # Update.
                inner_optimizer.step()
     
        prog_bar.set_description(f'MSE: {mse}')

        modul_np = modulator.detach().cpu().contiguous().numpy()
        update_dict = {
            'modul': modul_np,
            'label': label[0].item(),
            'is_spatial': is_spatial,
            'phi_shape': tuple(modul_np.shape),
        }
        if voxels:
            update_dict.update({'n_pts': int(image.sum().item())})

        functaset.append(update_dict)

    return functaset


# Split the train, validation and test functaset.
def split(functaset,name="functaset", ratio=(0.8,0.2), root="."):
    #functaset for splitting, name for file saving, ration in (train,val) format
    assignment = torch.tensor([0] * (int(len(functaset)*ratio[0])) + [1] * (int(len(functaset)*ratio[1])))
    if len(assignment) != len(functaset): #due to int flooring
        assignment = torch.cat((assignment,torch.Tensor([1]*(abs(len(functaset)-len(assignment))))))
    assignment = assignment[torch.randperm(len(functaset))]
    train_set, val_set = [], []
    for i in range(len(functaset)):
        if assignment[i] == 0:
            train_set.append(functaset[i])
        else:
            val_set.append(functaset[i])
        
    
    os.makedirs(f'{root}/functaset', exist_ok=True)
    joblib.dump(train_set, f'{root}/functaset/{name}_train.pkl')
    joblib.dump(val_set, f'{root}/functaset/{name}_val.pkl')
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--lr', type=float, default=0.01, help='per-sample optimization lr')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--depth', type=int, default=10, help='SIREN depth')
    parser.add_argument('--inr-type', choices=['siren', 'fourier_siren', 'finer', 'fourier_lsa'], default='siren',
                        help='Coordinate INR backbone type. Overridden by checkpoint.model_args if present.')
    parser.add_argument('--lsa-num-harmonics', type=int, default=8,
                        help='Number of harmonics K in learnable spectral activation.')

    parser.add_argument('--lsa-init-scale', type=float, default=1e-3,
                       help='Initial std scale for learnable spectral activation coefficients.')

    parser.add_argument('--lsa-no-linear', action='store_true', default=False,
                        help='If set, remove the identity term u from LSA activation.')

    parser.add_argument('--inner-optim', choices=['sgd', 'adam'], default='sgd',
                    help='Optimizer for fitting phi when creating the functaset.')
    parser.add_argument('--fourier-num-freqs', type=int, default=64,
                        help='Number of random Fourier frequencies for --inr-type fourier_siren.')
    parser.add_argument('--fourier-sigma', type=float, default=10.0,
                        help='Stddev of Gaussian Fourier frequency matrix B.')
    parser.add_argument('--fourier-include-input', action='store_true', default=False,
                        help='Concatenate raw (x,y) coordinates to Fourier features.')
    parser.add_argument('--freq', type=float, default=30.0,
                        help='ω0 used by SIREN, FINER, and Fourier-SIREN hidden layers. '
                             'Overridden by checkpoint model_args.freq if present.')
    parser.add_argument('--finer-first-bias-scale', type=float, default=1.0,
                        help='FINER first-layer bias init range: U(-scale, scale). '
                             'This controls the supported frequency set.')
    parser.add_argument('--finer-scale-req-grad', action='store_true', default=False,
                        help='If set, allow gradients through FINER scale = |z| + 1. '
                             'Default false is closer to the reference implementation.')
    parser.add_argument('--dataset', choices=["mnist", "fmnist", "cifar10", "modelnet"], help="Train for MNIST, Fashion-MNIST, CIFAR-10, or ModelNet10")
    parser.add_argument('--iters', type=int, default=100, help='number of optimization iterations per sample')
    parser.add_argument('--data-path', type=str, default='..', help='path to MNIST,FMNIST or ModelNet10 dataset')
    parser.add_argument('--checkpoint', type=str, help='path to pretrained SIREN from meta-optimization')
    parser.add_argument('--lbfgs', action='store_true', default=False, help="whether to use L-BFGS or SGD/Adam optimization.")
    parser.add_argument('--saveroot', type=str, default=".", help='root save dir to save functasets')
    parser.add_argument(
        '--functaset-stem', type=str, default=None,
        help='Base name for functaset pickles under saveroot/functaset/: '
             '{stem}_train.pkl, {stem}_val.pkl, {stem}_test.pkl. '
             'Defaults to the dataset name (e.g. cifar10).',
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    parser.add_argument('--variant', choices=variants.available(), default='vanilla',
                        help='SIREN variant used at training time. Must match the '
                             'variant recorded in the checkpoint so architecture '
                             'wrappers (if any) line up before loading state_dict.')

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    # Spatial Functa flags. Values present in checkpoint.model_args take precedence.
    parser.add_argument('--spatial-modulation', action='store_true', default=False,
                        help='Use a spatial latent grid (Spatial Functa). Defaults '
                             'are overridden by checkpoint.model_args if present.')
    parser.add_argument('--latent-spatial-dim', type=int, default=8,
                        help='Spatial side s of the latent grid. Used only when '
                             '--spatial-modulation is set or the checkpoint is spatial.')
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Per-cell latent channels c of the latent grid.')
    parser.add_argument('--spatial-interp', choices=['nearest'], default='nearest',
                        help='Spatial latent interpolation; only nearest (1-NN) is supported.')
    parser.add_argument('--use-local-coords', action='store_true', default=False,
                        help='Feed per-cell local coordinates instead of global coords.')
    parser.add_argument('--modulation-type', choices=['shift'], default='shift',
                        help='Spatial modulation type; only shift is supported.')

    variants.add_all_variant_args(parser)
    return parser.parse_args()


def _model_args_from_checkpoint(args, ckpt):
    model_args = {
        'dataset': args.dataset,
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
        'freq': args.freq,
        'finer_first_bias_scale': args.finer_first_bias_scale,
        'finer_scale_req_grad': args.finer_scale_req_grad,
        'lsa_num_harmonics': args.lsa_num_harmonics,
        'lsa_init_scale': args.lsa_init_scale,
        'lsa_include_linear': not args.lsa_no_linear,
        'spatial_modulation': bool(getattr(args, 'spatial_modulation', False)),
        'latent_spatial_dim': args.latent_spatial_dim,
        'latent_dim': args.latent_dim,
        'spatial_interp': args.spatial_interp,
        'use_local_coords': bool(args.use_local_coords),
        'modulation_type': args.modulation_type,
    }
    ckpt_model_args = ckpt.get('model_args', {}) or {}
    for key in model_args:
        if key in ckpt_model_args:
            model_args[key] = ckpt_model_args[key]
    return model_args


def _build_2d_model(model_args, device):
    inr_type = model_args.get('inr_type', 'siren')

    if model_args.get('spatial_modulation', False):
        model = SpatialModulatedINR(
            height=model_args['height'],
            width=model_args['width'],
            hidden_features=model_args['hidden_dim'],
            num_layers=model_args['depth'],
            latent_spatial_dim=model_args.get('latent_spatial_dim', 8),
            latent_dim=model_args.get('latent_dim', 16),
            base_inr_type=inr_type,
            spatial_interp=model_args.get('spatial_interp', 'nearest'),
            use_local_coords=model_args.get('use_local_coords', True),
            modulation_type=model_args.get('modulation_type', 'shift'),
            freq=model_args.get('freq', 30.0),
            device=device,
            out_features=model_args['out_features'],
            fourier_num_freqs=model_args.get('fourier_num_freqs', 64),
            fourier_sigma=model_args.get('fourier_sigma', 10.0),
            fourier_include_input=model_args.get('fourier_include_input', False),
            first_bias_scale=model_args.get('finer_first_bias_scale', 1.0),
            scale_req_grad=model_args.get('finer_scale_req_grad', False),
            lsa_num_harmonics=model_args.get('lsa_num_harmonics', 8),
            lsa_init_scale=model_args.get('lsa_init_scale', 1e-3),
            lsa_include_linear=model_args.get('lsa_include_linear', True),
        )
        print(f"[build] {type(model).__name__}  base_inr_type={model.base_inr_type}  "
              f"is_spatial={model.is_spatial}  phi_shape={tuple(model.phi_shape)}  "
              f"phi_numel={int(model.phi_numel)}")
        return model

    if inr_type == 'fourier_siren':
        model = ModulatedFourierSIREN(
            height=model_args['height'],
            width=model_args['width'],
            hidden_features=model_args['hidden_dim'],
            num_layers=model_args['depth'],
            modul_features=model_args['mod_dim'],
            device=device,
            out_features=model_args['out_features'],
            freq=model_args.get('freq', 30.0),
            fourier_num_freqs=model_args.get('fourier_num_freqs', 64),
            fourier_sigma=model_args.get('fourier_sigma', 10.0),
            fourier_include_input=model_args.get('fourier_include_input', False),
        )
    elif inr_type == 'fourier_lsa':
        model = ModulatedFourierLSA(
            height=model_args['height'],
            width=model_args['width'],
            hidden_features=model_args['hidden_dim'],
            num_layers=model_args['depth'],
            modul_features=model_args['mod_dim'],
            device=device,
            out_features=model_args['out_features'],
            fourier_num_freqs=model_args.get('fourier_num_freqs', 64),
            fourier_sigma=model_args.get('fourier_sigma', 10.0),
            fourier_include_input=model_args.get('fourier_include_input', False),
            lsa_num_harmonics=model_args.get('lsa_num_harmonics', 8),
            lsa_init_scale=model_args.get('lsa_init_scale', 1e-3),
            lsa_include_linear=model_args.get('lsa_include_linear', True),
        )
    elif inr_type == 'finer':
        model = ModulatedFINER(
            height=model_args['height'],
            width=model_args['width'],
            hidden_features=model_args['hidden_dim'],
            num_layers=model_args['depth'],
            modul_features=model_args['mod_dim'],
            device=device,
            out_features=model_args['out_features'],
            freq=model_args.get('freq', 30.0),
            first_bias_scale=model_args.get('finer_first_bias_scale', 1.0),
            scale_req_grad=model_args.get('finer_scale_req_grad', False),
        )
    else:  # 'siren' or unknown -> default SIREN
        model = ModulatedSIREN(
            height=model_args['height'],
            width=model_args['width'],
            hidden_features=model_args['hidden_dim'],
            num_layers=model_args['depth'],
            modul_features=model_args['mod_dim'],
            device=device,
            out_features=model_args['out_features'],
            freq=model_args.get('freq', 30.0),
        )

    print(f"[build] {type(model).__name__}  is_spatial={model.is_spatial}  "
          f"phi_shape={tuple(model.phi_shape)}  phi_numel={int(model.phi_numel)}")
    return model



def maybe_limit_dataloader(dataloader, max_samples, split_name):
    if max_samples is None:
        print(f"[makeset] {split_name} full size: {len(dataloader.dataset)}")
        return dataloader

    max_samples = min(max_samples, len(dataloader.dataset))

    subset = Subset(
        dataloader.dataset,
        range(max_samples)
    )

    print(f"[makeset] {split_name} subset size: {len(subset)}")

    return DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=getattr(dataloader, "num_workers", 0),
        pin_memory=getattr(dataloader, "pin_memory", False),
    )
    
if __name__ == '__main__':

    args = get_args()
    set_random_seeds(args.seed, args.device)

    pretrained = torch.load(args.checkpoint, map_location=args.device)
    model_args = _model_args_from_checkpoint(args, pretrained)

    if args.dataset == "modelnet":
        if model_args.get('inr_type', 'siren') != 'siren':
            raise SystemExit("fourier_siren/finer/fourier_lsa are currently supported for 2D image datasets only.")

        resample_shape = (15, 15, 15)  # we use this resampling in all experiments

        dataloader_train = get_modelnet_loader(
            train=True,
            batch_size=1,
            resample_shape=resample_shape,
        )

        dataloader_test = get_modelnet_loader(
            train=False,
            batch_size=1,
            resample_shape=resample_shape,
        )

        modSiren = ModulatedSIREN3D(
            height=resample_shape[0],
            width=resample_shape[1],
            depth=resample_shape[2],
            hidden_features=model_args['hidden_dim'],
            num_layers=model_args['depth'],
            modul_features=model_args['mod_dim'],
        )

    else:
        if args.dataset == "cifar10":

            dataloader_train = get_cifar10_loader(
                args.data_path,
                train=True,
                batch_size=1,
            )

            dataloader_test = get_cifar10_loader(
                args.data_path,
                train=False,
                batch_size=1,
            )

            modSiren = _build_2d_model(model_args, args.device)

        else:
            dataloader_train = get_mnist_loader(
                args.data_path,
                train=True,
                batch_size=1,
                fashion=args.dataset == "fmnist",
            )

            dataloader_test = get_mnist_loader(
                args.data_path,
                train=False,
                batch_size=1,
                fashion=args.dataset == "fmnist",
            )

            modSiren = _build_2d_model(model_args, args.device)

    # ---------------------------------------------------------
    # Optional subset slicing for faster functaset creation.
    # Important: do NOT assign dataloader.dataset after DataLoader
    # is initialized. Instead, create a new limited DataLoader.
    # ---------------------------------------------------------

    dataloader_train = maybe_limit_dataloader(
        dataloader_train,
        args.max_train_samples,
        "train",
    )

    dataloader_test = maybe_limit_dataloader(
        dataloader_test,
        args.max_test_samples,
        "test",
    )

    modSiren = variants.build(args.variant, modSiren, args)
    modSiren.load_state_dict(pretrained['state_dict'])

    print("[makeset] checkpoint:", args.checkpoint)
    print("[makeset] model_args:", model_args)
    print("[makeset] built model:", type(modSiren).__name__)
    print("[makeset] modul_features:", getattr(modSiren, "modul_features", None))
    print("[makeset] hidden_features:", getattr(getattr(modSiren, "siren", None), "hidden_features", None))
    print("[makeset] coord_normalization:", model_args.get("coord_normalization"))

    functaset_stem = args.functaset_stem if args.functaset_stem is not None else args.dataset

    functa_trainset = create_functaset(
        modSiren,
        dataloader_train,
        inner_steps=args.iters,
        inner_lr=args.lr,
        voxels=args.dataset == "modelnet",
        lbfgs=args.lbfgs,
        inner_optim=args.inner_optim,
    )

    split(
        functa_trainset,
        name=functaset_stem,
        root=args.saveroot,
    )

    functa_testset = create_functaset(
        modSiren,
        dataloader_test,
        inner_steps=args.iters,
        inner_lr=args.lr,
        voxels=args.dataset == "modelnet",
        lbfgs=args.lbfgs,
        inner_optim=args.inner_optim,
    )

    joblib.dump(
        functa_testset,
        f'{args.saveroot}/functaset/{functaset_stem}_test.pkl',
    )