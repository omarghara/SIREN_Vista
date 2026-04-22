import os
import os.path as osp
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import get_mnist_functa, get_mnist_loader
from utils import adjust_learning_rate, set_random_seeds, get_accuracy, Average
from tqdm import tqdm
import numpy as np
import argparse
from SIREN import ModulatedSIREN
from train_classifier import Classifier
from matplotlib import pyplot as plt
from higher import get_diff_optim


class FullPGD(nn.Module):
    def __init__(self, inr, classifier, inner_steps = 100, inner_lr = 0.01, device='cuda'):
        """
        :param inr: pretrained inr model
        :param classifier: 'clean' pretrained classifier model to attack
        :param inner_steps: number of modulation optimization steps.
        :param inner_lr: learn rate for internal modulation optimization.
        :param device: use cuda or cpu.
        """
        super(FullPGD, self).__init__()
        #load classifier
        self.classifier = classifier.to(device)
        self.classifier.eval()
        
        #load inr
        self.inr = inr.to(device)
        self.inr.eval()
        
        #optimization params
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
       
    
    def fit_image(self, x, start_mod, clean, return_mse = False):
        '''optimize and return modulation for a specific (potentially perturbed) signal-domain input x'''
        device = "cuda"
        inner_criterion = nn.MSELoss().cuda()
        
        
        mses = []
        mods = None
        # Inner Optimization.
        for image in x:
            modulator = start_mod.squeeze().detach().clone().float().to(device) if start_mod is not None else torch.zeros(self.inr.modul_features).float().to(device)
         
            modulator.requires_grad = True
            image = image[0].view(1, -1).T.to(device)
            
            inner_optimizer = torch.optim.SGD([modulator],lr=self.inner_lr) if clean else get_diff_optim(optim.SGD([modulator], lr=self.inner_lr), [modulator], device='cuda')
            
            mse = 0
            for step in range(self.inner_steps):
                
                if clean:
                    inner_optimizer.zero_grad()
                    fitted = self.inr(modulator)
                    inner_loss = inner_criterion(fitted.flatten(), image.flatten())
                    mse = inner_loss.item()
                    inner_loss.backward()
                    # Clip the gradient.
                    torch.nn.utils.clip_grad_norm_([modulator], 1)
                    # Update.
                    inner_optimizer.step()
                    
                    
                else:
                    fitted = self.inr(modulator)
                    inner_loss = inner_criterion(fitted.flatten(), image.flatten())
                    mse = inner_loss.item()
                    modulator, = inner_optimizer.step(inner_loss,params=[modulator])
                
         
            mses.append(mse)
            mods = modulator.unsqueeze(0) if mods is None else torch.cat((modulator.unsqueeze(0),mods),axis=0)
      
      
        return torch.flip(mods,dims=[0]) if not return_mse else (torch.flip(mods,dims=[0]), mse)
        
    
    
    def forward(self, x, start_mod = None, clean=False, return_mse = False):
        '''find modulation for x and classify it.
           x is input in signal domain, start_mod is optional starting point for modulation optimization, clean indicates if x is under clean (non-perturbed) evaluation 
           (so, if not, we can skip costly gradient tracking in modulation optimization), return_mse indicates whether to return the best modulation representation error.'''
    
        modulations = self.fit_image(x, start_mod,clean, return_mse)
       
        if return_mse:
            modulations,mse = modulations
     
        preds = self.classifier(modulations)
        if return_mse:
            return preds, modulations.detach(), mse
        else:
            return preds, modulations.detach()


    
def run_attack(model, loader, criterion, constraint, pgd_iters, pgd_lr,
               device='cuda', max_samples=0, output_json=None):
    """
    :param model: attack model holding pretrained INR and classifier.
    :param loader: signal domain data-loader for data for which we optimize attacks.
    :param criterion: clean classifier optimization criterion.
    :param constraint: l_inf attack bound.
    :param pgd_iters: number of PGD iterations.
    :param pgd_lr: learn-rate for PGD attack.
    :param device: use cuda or cpu.
    :param max_samples: stop after this many samples (0 = run full loader).
    :param output_json: if set, write a JSON summary to this path.
    """
    total = len(loader) if not max_samples else min(max_samples, len(loader))
    prog_bar = tqdm(loader, total=total)
    rights_clean = 0        # clean predictions correct
    rights_attacked = 0     # robust predictions correct (unconditional)
    rights_both = 0         # clean AND robust both correct (conditional numerator)
    samples = 0

    for x, labels in prog_bar:
        if max_samples and samples >= max_samples:
            break
        samples += 1
        x = x.cuda()
        labels = labels.cuda()

        clean, clean_mod, clean_mse = model(x.unsqueeze(1), clean=True, return_mse=True)
        clean_correct = (clean.argmax(1) == labels).item()
        rights_clean += clean_correct
        labels = labels.cuda()

        pert = torch.zeros(28, 28).to(device)
        pert.requires_grad = True
        optimizer = torch.optim.Adam([pert], lr=pgd_lr)

        for iter in range(pgd_iters):
            optimizer.zero_grad()
            proj_pert = torch.clamp(pert, -constraint, constraint)
            output, cur_mod = model((x + proj_pert).unsqueeze(1), clean_mod, clean=False)
            loss = 1 - criterion(output, labels)
            loss.backward()
            optimizer.step()

        # Final evaluation on the perturbed input
        output, cur_mod, final_mse = model(
            (x + proj_pert).detach().unsqueeze(1), clean_mod.detach(),
            clean=True, return_mse=True,
        )

        print(f"Clean MSE: {clean_mse}. Perturbation Final MSE: {final_mse}")
        with torch.no_grad():
            attack_correct = (output.argmax(1) == labels).item()
            if attack_correct:
                rights_attacked += 1
            if attack_correct and clean_correct:
                rights_both += 1

        clean_acc = rights_clean / samples
        robust_acc = rights_attacked / samples
        cond_robust = rights_both / rights_clean if rights_clean > 0 else 0.0
        prog_bar.set_description(
            f'eps={constraint:.4f}: n={samples} clean={clean_acc:.3f} '
            f'robust={robust_acc:.3f} robust|clean={cond_robust:.3f}'
        )

    prog_bar.close()

    clean_acc = rights_clean / samples if samples else 0.0
    robust_acc = rights_attacked / samples if samples else 0.0
    cond_robust = rights_both / rights_clean if rights_clean else 0.0

    print()
    print(f"[final] eps={constraint:.6f} n={samples}")
    print(f"[final]   clean correct           : {rights_clean}/{samples}  = {clean_acc:.4f}")
    print(f"[final]   robust correct (uncond.): {rights_attacked}/{samples}  = {robust_acc:.4f}")
    print(f"[final]   robust|clean (condit.)  : {rights_both}/{rights_clean}  = {cond_robust:.4f}")

    if output_json:
        rec = {
            'constraint': constraint,
            'pgd_iters': pgd_iters,
            'pgd_lr': pgd_lr,
            'n_samples': samples,
            'clean_correct': rights_clean,
            'robust_correct_unconditional': rights_attacked,
            'both_correct': rights_both,
            'clean_acc': clean_acc,
            'robust_acc': robust_acc,
            'conditional_robust_acc': cond_robust,
        }
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(rec, f, indent=2)
        print(f"[final] wrote {output_json}")

    return {
        'n_samples': samples,
        'clean_acc': clean_acc,
        'robust_acc': robust_acc,
        'conditional_robust_acc': cond_robust,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--inner-lr', type=float, default=0.01, help='learn rate for internal modulation optimization')
    parser.add_argument('--ext-lr', type=float, default=0.01, help='learn rate for external adversarial perturbation optimization')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--mod-steps', type=int, default=10, help='Number of internal modulation optimization steps per PGD iteration')
    parser.add_argument('--pgd-steps', type=int, default=100, help='Number of projected gradient descent steps')
    parser.add_argument('--cwidth', type=int, default=512, help='classifier MLP hidden dimension')
    parser.add_argument('--cdepth', type=int, default=3, help='classifier MLP depth')
    parser.add_argument('--depth', type=int, default=10, help='SIREN depth')
    parser.add_argument('--dataset', choices=["mnist", "fmnist"], help="Train for MNIST or Fashion-MNIST") #We currently do not support Full-PGD for ModelNet10
    parser.add_argument('--data-path', type=str, default='..', help='path to MNIST or FMNIST dataset')
    parser.add_argument('--siren-checkpoint', type=str, help='path to pretrained SIREN from meta-optimization')
    parser.add_argument('--classifier-checkpoint', type=str, help='path to pretrained classifier')
    parser.add_argument('--epsilon', type=int, default=16, help='attack epsilon -- epsilon/255 is the de-facto attack l_inf constraint.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    parser.add_argument('--max-samples', type=int, default=0, help='Stop after this many samples (0 = full test set).')
    parser.add_argument('--output-json', type=str, default=None, help='Optional path to write a JSON summary of final metrics.')
    return parser.parse_args()
    
if __name__ == '__main__':

    
    args = get_args()    
    set_random_seeds(args.seed, args.device)
         
    dataloader = get_mnist_loader(args.data_path, train=False, batch_size=1, fashion = args.dataset=="fmnist")
    
    #Initiallize pretrained models
    modSiren = ModulatedSIREN(height=28, width=28, hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #28,28 is mnist and fmnist dims
    pretrained = torch.load(args.siren_checkpoint)
    modSiren.load_state_dict(pretrained['state_dict'])
    
    classifier = Classifier(width=args.cwidth, depth=args.cdepth,
                            in_features=args.mod_dim, num_classes=10).to(args.device)
    pretrained = torch.load(args.classifier_checkpoint)
    classifier.load_state_dict(pretrained['state_dict'])
    classifier.eval()
        
    attack_model = FullPGD(modSiren, classifier, inner_steps=args.mod_steps, inner_lr = args.inner_lr, device=args.device)
    attack_model.to(args.device)
    constraint = args.epsilon/255
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    run_attack(attack_model, dataloader, criterion, constraint, args.pgd_steps, args.ext_lr, args.device,
               max_samples=args.max_samples, output_json=args.output_json)