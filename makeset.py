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
        lbfgs=False
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
    modul_features = model.modul_features
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))
    
    for image, label in prog_bar:
        
        image = image.squeeze().to(device) if voxels else _prep_2d_image(image[0], device)
        modulator = torch.zeros(modul_features).float().to(device)
        modulator.requires_grad = True
        
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
            inner_optimizer = (optim.Adam if voxels else optim.SGD)([modulator], lr=inner_lr)
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
        
        update_dict = {'modul': modulator.detach().cpu().numpy(),
             'label': label[0].item()}
        if voxels:
            update_dict.update({'n_pts':int(image.sum().item())})
        
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
        
    
    os.makedirs(f'{root}/functaset')
    joblib.dump(train_set, f'{root}/functaset/{name}_train.pkl')
    joblib.dump(val_set, f'{root}/functaset/{name}_val.pkl')
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--lr', type=float, default=0.01, help='per-sample optimization lr')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--depth', type=int, default=10, help='SIREN depth')
    parser.add_argument('--inr-type', choices=['siren', 'fourier_siren'], default='siren',
                        help='Coordinate INR backbone type. Overridden by checkpoint.model_args if present.')
    parser.add_argument('--fourier-num-freqs', type=int, default=64,
                        help='Number of random Fourier frequencies for --inr-type fourier_siren.')
    parser.add_argument('--fourier-sigma', type=float, default=10.0,
                        help='Stddev of Gaussian Fourier frequency matrix B.')
    parser.add_argument('--fourier-include-input', action='store_true', default=False,
                        help='Concatenate raw (x,y) coordinates to Fourier features.')
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
    }
    ckpt_model_args = ckpt.get('model_args', {}) or {}
    for key in model_args:
        if key in ckpt_model_args:
            model_args[key] = ckpt_model_args[key]
    return model_args


def _build_2d_model(model_args, device):
    if model_args.get('inr_type', 'siren') == 'fourier_siren':
        return ModulatedFourierSIREN(
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
        )
    return ModulatedSIREN(
        height=model_args['height'],
        width=model_args['width'],
        hidden_features=model_args['hidden_dim'],
        num_layers=model_args['depth'],
        modul_features=model_args['mod_dim'],
        device=device,
        out_features=model_args['out_features'],
    )


if __name__ == '__main__':

    
    args = get_args()    
    set_random_seeds(args.seed, args.device)
    
    pretrained = torch.load(args.checkpoint, map_location=args.device)
    model_args = _model_args_from_checkpoint(args, pretrained)

    if args.dataset == "modelnet":
        if model_args.get('inr_type', 'siren') != 'siren':
            raise SystemExit("fourier_siren is currently supported for 2D image datasets only.")
        resample_shape = (15,15,15) #we use this resampling in all experiments
        dataloader_train = get_modelnet_loader(train=True, batch_size=1, resample_shape=resample_shape)
        dataloader_test = get_modelnet_loader(train=False, batch_size=1, resample_shape=resample_shape)
        modSiren = ModulatedSIREN3D(height=resample_shape[0], width=resample_shape[1], depth=resample_shape[2],\
            hidden_features=model_args['hidden_dim'], num_layers=model_args['depth'], modul_features=model_args['mod_dim']) #we use a mod dim of 2048 in our exps
  
    else:
        if args.dataset == "cifar10":
            dataloader_train = get_cifar10_loader(args.data_path, train=True, batch_size=1)
            dataloader_test = get_cifar10_loader(args.data_path, train=False, batch_size=1)
            modSiren = _build_2d_model(model_args, args.device)
        else:
            dataloader_train = get_mnist_loader(args.data_path, train=True, batch_size=1, fashion = args.dataset=="fmnist")
            dataloader_test = get_mnist_loader(args.data_path, train=False, batch_size=1, fashion = args.dataset=="fmnist")
            modSiren = _build_2d_model(model_args, args.device) #28,28 is mnist and fmnist dims

    modSiren = variants.build(args.variant, modSiren, args)
    modSiren.load_state_dict(pretrained['state_dict'])
    functa_trainset = create_functaset(modSiren, dataloader_train, inner_steps=args.iters, inner_lr=args.lr, voxels=args.dataset=="modelnet", lbfgs=args.lbfgs)
    functaset_stem = args.functaset_stem if args.functaset_stem is not None else args.dataset
    split(functa_trainset, name=functaset_stem, root=args.saveroot) #Split to training, validation and save split functaset
    functa_testset = create_functaset(modSiren, dataloader_test, inner_steps=args.iters, inner_lr=args.lr, voxels=args.dataset=="modelnet",lbfgs=args.lbfgs)
    joblib.dump(functa_testset, f'{args.saveroot}/functaset/{functaset_stem}_test.pkl')
    

