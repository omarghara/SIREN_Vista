import argparse
import json
import os
import sys
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim
from higher import get_diff_optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import variants
from dataloader import get_cifar10_loader
from SIREN import (
    ModulatedFINER,
    ModulatedFourierLSA,
    ModulatedFourierSIREN,
    ModulatedSIREN,
    SpatialModulatedINR,
)
from train_classifier import (
    Classifier,
    SpatialFunctaTransformer,
    SpatialPhiCNN,
    _prepare_modulations,
)
from utils import set_random_seeds


def _variant_args_from_checkpoint(ckpt):
    variant_args = ckpt.get("variant_args", {}) or {}
    return Namespace(
        soft_lip_cap=variant_args.get("soft_lip_cap", 1.0),
        soft_lip_lambda=variant_args.get("soft_lip_lambda", 1e-2),
        soft_lip_apply_to=variant_args.get("soft_lip_apply_to", "sine_and_readout"),
        soft_lip_power_iters=variant_args.get("soft_lip_power_iters", 1),
        soft_lip_skip_first=variant_args.get("soft_lip_skip_first", False),
    )


def _build_inr_from_args(model_args, state, device):
    hidden_dim = model_args.get("hidden_dim", 256)
    depth = model_args.get("depth", 6)
    mod_dim = model_args.get("mod_dim", 1024)
    height = model_args.get("height", 32)
    width = model_args.get("width", 32)
    out_features = model_args.get("out_features", 3)

    inr_type = model_args.get("inr_type", "siren")
    if inr_type == "siren" and "fourier.B" in state:
        inr_type = "fourier_siren"

    is_spatial = bool(model_args.get("spatial_modulation", False)) or bool(
        model_args.get("is_spatial", False)
    )
    if is_spatial:
        return SpatialModulatedINR(
            height=height,
            width=width,
            hidden_features=hidden_dim,
            num_layers=depth,
            latent_spatial_dim=model_args.get("latent_spatial_dim", 8),
            latent_dim=model_args.get("latent_dim", 16),
            base_inr_type=inr_type,
            spatial_interp=model_args.get("spatial_interp", "nearest"),
            use_local_coords=bool(model_args.get("use_local_coords", True)),
            modulation_type=model_args.get("modulation_type", "shift"),
            freq=model_args.get("freq", 30.0),
            device=device,
            out_features=out_features,
            fourier_num_freqs=model_args.get("fourier_num_freqs", 64),
            fourier_sigma=model_args.get("fourier_sigma", 10.0),
            fourier_include_input=bool(model_args.get("fourier_include_input", False)),
            first_bias_scale=model_args.get("finer_first_bias_scale", None),
            scale_req_grad=bool(model_args.get("finer_scale_req_grad", False)),
            lsa_num_harmonics=model_args.get("lsa_num_harmonics", 8),
            lsa_init_scale=model_args.get("lsa_init_scale", 1e-3),
            lsa_include_linear=bool(model_args.get("lsa_include_linear", True)),
        ).to(device)

    if inr_type == "fourier_siren":
        nf = model_args.get("fourier_num_freqs")
        if nf is None:
            nf = int(state["fourier.B"].shape[0]) if "fourier.B" in state else 64
        return ModulatedFourierSIREN(
            height=height,
            width=width,
            hidden_features=hidden_dim,
            num_layers=depth,
            modul_features=mod_dim,
            device=device,
            out_features=out_features,
            freq=model_args.get("freq", 30.0),
            fourier_num_freqs=nf,
            fourier_sigma=model_args.get("fourier_sigma", 10.0),
            fourier_include_input=bool(model_args.get("fourier_include_input", False)),
        ).to(device)

    if inr_type == "fourier_lsa":
        nf = model_args.get("fourier_num_freqs")
        if nf is None:
            nf = int(state["fourier.B"].shape[0]) if "fourier.B" in state else 64
        return ModulatedFourierLSA(
            height=height,
            width=width,
            hidden_features=hidden_dim,
            num_layers=depth,
            modul_features=mod_dim,
            device=device,
            out_features=out_features,
            fourier_num_freqs=nf,
            fourier_sigma=model_args.get("fourier_sigma", 10.0),
            fourier_include_input=bool(model_args.get("fourier_include_input", False)),
            lsa_num_harmonics=model_args.get("lsa_num_harmonics", 8),
            lsa_init_scale=model_args.get("lsa_init_scale", 1e-3),
            lsa_include_linear=bool(model_args.get("lsa_include_linear", True)),
        ).to(device)

    if inr_type == "finer":
        return ModulatedFINER(
            height=height,
            width=width,
            hidden_features=hidden_dim,
            num_layers=depth,
            modul_features=mod_dim,
            device=device,
            out_features=out_features,
            freq=model_args.get("finer_freq", model_args.get("freq", 30.0)),
            first_bias_scale=model_args.get("finer_first_bias_scale", 1.0),
            scale_req_grad=bool(model_args.get("finer_scale_req_grad", False)),
        ).to(device)

    return ModulatedSIREN(
        height=height,
        width=width,
        hidden_features=hidden_dim,
        num_layers=depth,
        modul_features=mod_dim,
        device=device,
        out_features=out_features,
        freq=model_args.get("freq", 30.0),
    ).to(device)


def load_inr_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model_args = ckpt.get("model_args", {}) or {}
    state = ckpt.get("state_dict", {}) or {}
    model = _build_inr_from_args(model_args, state, device)
    variant = ckpt.get("variant", "vanilla")
    model = variants.build(variant, model, _variant_args_from_checkpoint(ckpt))
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def _detect_mlp_arch(state):
    linear_indices = sorted(
        int(k.split(".")[1])
        for k in state
        if k.startswith("net.") and k.endswith(".weight") and state[k].dim() == 2
    )
    max_idx = max(linear_indices)
    depth = max_idx // 3
    width = state["net.1.weight"].shape[0]
    in_features = state["net.1.weight"].shape[1]
    num_classes = state[f"net.{max_idx}.weight"].shape[0]
    return depth, width, in_features, num_classes


def load_classifier_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    classifier_type = ckpt.get("classifier_type", "mlp")
    latent_spatial_dim = ckpt.get("latent_spatial_dim", None)
    latent_dim = ckpt.get("latent_dim", None)
    num_classes = 10

    if classifier_type == "mlp":
        if "cwidth" in ckpt and "cdepth" in ckpt:
            width = int(ckpt["cwidth"])
            depth = int(ckpt["cdepth"])
            in_features = int(ckpt.get("mod_dim", state["net.1.weight"].shape[1]))
        else:
            depth, width, in_features, num_classes = _detect_mlp_arch(state)
        classifier = Classifier(
            width=width,
            depth=depth,
            in_features=in_features,
            num_classes=num_classes,
            dropout=float(ckpt.get("dropout", 0.0)),
        ).to(device)
    elif classifier_type == "cnn":
        if latent_dim is None:
            raise RuntimeError("CNN classifier checkpoint is missing 'latent_dim'.")
        classifier = SpatialPhiCNN(
            in_channels=int(latent_dim),
            num_classes=num_classes,
            width=int(ckpt.get("cnn_width", 128)),
            dropout=float(ckpt.get("dropout", 0.1)),
        ).to(device)
    elif classifier_type == "vit":
        if latent_dim is None or latent_spatial_dim is None:
            raise RuntimeError("ViT classifier checkpoint is missing spatial metadata.")
        classifier = SpatialFunctaTransformer(
            latent_dim=int(latent_dim),
            spatial_dim=int(latent_spatial_dim),
            num_classes=num_classes,
            width=int(ckpt.get("vit_width", 128)),
            ffw_width=int(ckpt.get("vit_ffw_width", 256)),
            depth=int(ckpt.get("vit_depth", 12)),
            num_heads=int(ckpt.get("vit_heads", 16)),
            dropout=float(ckpt.get("dropout", 0.1)),
            normalizing_factor=float(ckpt.get("vit_normalizing_factor", 0.08)),
        ).to(device)
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type!r}")

    classifier.load_state_dict(state)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    classifier._clf_type = classifier_type
    classifier._latent_spatial_dim = (
        int(latent_spatial_dim) if latent_spatial_dim is not None else None
    )
    classifier._latent_dim = int(latent_dim) if latent_dim is not None else None
    classifier._phi_mean = (
        ckpt["phi_mean"].to(device) if ckpt.get("phi_mean", None) is not None else None
    )
    classifier._phi_std = (
        ckpt["phi_std"].to(device).clamp_min(1e-6)
        if ckpt.get("phi_std", None) is not None
        else None
    )
    return classifier, ckpt


def apply_classifier(classifier, phi):
    x = _prepare_modulations(
        phi,
        classifier_type=getattr(classifier, "_clf_type", "mlp"),
        latent_spatial_dim=getattr(classifier, "_latent_spatial_dim", None),
        latent_dim=getattr(classifier, "_latent_dim", None),
    )
    if classifier._phi_mean is not None and classifier._phi_std is not None:
        x = (x - classifier._phi_mean) / classifier._phi_std
    return classifier(x)


def check_model_classifier_compat(inr, classifier):
    clf_type = getattr(classifier, "_clf_type", "mlp")
    if clf_type == "mlp":
        if classifier.in_features != inr.modul_features:
            raise RuntimeError(
                f"MLP classifier in_features={classifier.in_features} "
                f"does not match INR modul_features={inr.modul_features}."
            )
        return

    if not getattr(inr, "is_spatial", False):
        raise RuntimeError(f"{clf_type} classifier requires a spatial INR.")

    if classifier._latent_spatial_dim != inr.latent_spatial_dim:
        raise RuntimeError(
            f"Classifier spatial_dim={classifier._latent_spatial_dim} "
            f"does not match INR spatial_dim={inr.latent_spatial_dim}."
        )
    if classifier._latent_dim != inr.latent_dim:
        raise RuntimeError(
            f"Classifier latent_dim={classifier._latent_dim} "
            f"does not match INR latent_dim={inr.latent_dim}."
        )


class FullPGDCIFAR10Spatial(nn.Module):
    def __init__(
        self,
        inr,
        classifier,
        inner_steps=10,
        inner_lr=0.01,
        clean_grad_clip=0.0,
        device="cuda",
    ):
        super().__init__()
        self.inr = inr.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.inner_steps = int(inner_steps)
        self.inner_lr = float(inner_lr)
        self.clean_grad_clip = float(clean_grad_clip)
        self.device = device
        self.inner_criterion = nn.MSELoss().to(device)

    def _initial_phi(self, start_mod, index=0):
        if start_mod is not None:
            return start_mod[index].detach().clone().float().to(self.device)
        if hasattr(self.inr, "init_phi"):
            return self.inr.init_phi(device=self.device).float()
        return torch.zeros(self.inr.modul_features, device=self.device).float()

    def _fit_single(self, image, start_mod=None, start_index=0, clean=False):
        modulator = self._initial_phi(start_mod, index=start_index)
        modulator.requires_grad_(True)

        target = image.to(self.device).float().permute(1, 2, 0).reshape(
            -1, image.shape[0]
        )
        inner_optimizer = (
            optim.SGD([modulator], lr=self.inner_lr)
            if clean
            else get_diff_optim(
                optim.SGD([modulator], lr=self.inner_lr),
                [modulator],
                device=self.device,
            )
        )

        mse = 0.0
        for _ in range(self.inner_steps):
            if clean:
                inner_optimizer.zero_grad()
                fitted = self.inr(modulator)
                inner_loss = self.inner_criterion(fitted, target)
                mse = float(inner_loss.item())
                inner_loss.backward()
                if self.clean_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([modulator], self.clean_grad_clip)
                inner_optimizer.step()
            else:
                fitted = self.inr(modulator)
                inner_loss = self.inner_criterion(fitted, target)
                mse = float(inner_loss.item())
                (modulator,) = inner_optimizer.step(inner_loss, params=[modulator])

        return modulator, mse

    def fit_image(self, x, start_mod=None, clean=False, return_mse=False):
        mods = []
        mses = []
        for idx, image in enumerate(x):
            modulator, mse = self._fit_single(
                image, start_mod=start_mod, start_index=idx, clean=clean
            )
            mods.append(modulator)
            mses.append(mse)
        modulations = torch.stack(mods, dim=0)
        if return_mse:
            return modulations, float(sum(mses) / max(len(mses), 1))
        return modulations

    def forward(self, x, start_mod=None, clean=False, return_mse=False):
        modulations = self.fit_image(
            x, start_mod=start_mod, clean=clean, return_mse=return_mse
        )
        if return_mse:
            modulations, mse = modulations
        preds = apply_classifier(self.classifier, modulations)
        if return_mse:
            return preds, modulations.detach(), mse
        return preds, modulations.detach()


def run_attack(
    model,
    loader,
    criterion,
    constraint,
    pgd_iters,
    pgd_lr,
    device="cuda",
    max_samples=0,
    output_json=None,
    model_slug=None,
    classifier_type=None,
):
    total = len(loader) if not max_samples else min(max_samples, len(loader))
    prog_bar = tqdm(loader, total=total)
    rights_clean = 0
    rights_attacked = 0
    rights_both = 0
    samples = 0

    for x, labels in prog_bar:
        if max_samples and samples >= max_samples:
            break
        samples += 1
        x = x.to(device).float()
        labels = labels.to(device).long()

        clean, clean_mod, clean_mse = model(x, clean=True, return_mse=True)
        clean_correct = (clean.argmax(1) == labels).item()
        rights_clean += clean_correct

        pert = torch.zeros_like(x[0], device=device, requires_grad=True)
        optimizer = optim.Adam([pert], lr=pgd_lr)
        proj_pert = torch.clamp(pert, -constraint, constraint)

        for _ in range(pgd_iters):
            optimizer.zero_grad()
            proj_pert = torch.clamp(pert, -constraint, constraint)
            attacked_x = torch.clamp(x + proj_pert.unsqueeze(0), 0.0, 1.0)
            output, _ = model(attacked_x, clean_mod, clean=False)
            loss = -criterion(output, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_x = torch.clamp(x + proj_pert.detach().unsqueeze(0), 0.0, 1.0)
        output, _, final_mse = model(
            final_x, clean_mod.detach(), clean=True, return_mse=True
        )

        print(f"Clean MSE: {clean_mse}. Perturbation Final MSE: {final_mse}")
        with torch.no_grad():
            attack_correct = (output.argmax(1) == labels).item()
            rights_attacked += int(attack_correct)
            rights_both += int(attack_correct and clean_correct)

        clean_acc = rights_clean / samples
        robust_acc = rights_attacked / samples
        cond_robust = rights_both / rights_clean if rights_clean > 0 else 0.0
        prog_bar.set_description(
            f"eps={constraint:.4f}: n={samples} clean={clean_acc:.3f} "
            f"robust={robust_acc:.3f} robust|clean={cond_robust:.3f}"
        )

    prog_bar.close()

    clean_acc = rights_clean / samples if samples else 0.0
    robust_acc = rights_attacked / samples if samples else 0.0
    cond_robust = rights_both / rights_clean if rights_clean else 0.0

    print()
    print(f"[final] eps={constraint:.6f} n={samples}")
    print(f"[final]   clean correct           : {rights_clean}/{samples}  = {clean_acc:.4f}")
    print(
        f"[final]   robust correct (uncond.): {rights_attacked}/{samples}  = {robust_acc:.4f}"
    )
    print(
        f"[final]   robust|clean (condit.)  : {rights_both}/{rights_clean}  = {cond_robust:.4f}"
    )

    rec = {
        "dataset": "cifar10",
        "model_slug": model_slug,
        "classifier_type": classifier_type,
        "constraint": constraint,
        "pgd_iters": pgd_iters,
        "pgd_lr": pgd_lr,
        "n_samples": samples,
        "clean_correct": rights_clean,
        "robust_correct_unconditional": rights_attacked,
        "both_correct": rights_both,
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "conditional_robust_acc": cond_robust,
    }
    if output_json:
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[final] wrote {output_json}")
    return rec


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--ext-lr", type=float, default=0.01)
    parser.add_argument("--mod-steps", type=int, default=10)
    parser.add_argument("--pgd-steps", type=int, default=100)
    parser.add_argument(
        "--clean-grad-clip",
        type=float,
        default=0.0,
        help=(
            "Optional grad-norm clip for non-differentiable clean/final phi "
            "fitting. Default 0 disables clipping to match makeset.py."
        ),
    )
    parser.add_argument("--data-path", type=str, default="../data")
    parser.add_argument("--siren-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument(
        "--epsilon",
        type=int,
        default=16,
        help="epsilon numerator; attack bound is epsilon / 255",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after this many test samples (0 = full test set).",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_random_seeds(args.seed, args.device)

    loader = get_cifar10_loader(
        args.data_path,
        train=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    inr, inr_ckpt = load_inr_checkpoint(args.siren_checkpoint, args.device)
    classifier, classifier_ckpt = load_classifier_checkpoint(
        args.classifier_checkpoint, args.device
    )
    check_model_classifier_compat(inr, classifier)

    print("[full-pgd-cifar10-spatial] loaded")
    print(f"  siren checkpoint     : {args.siren_checkpoint}")
    print(f"  classifier checkpoint: {args.classifier_checkpoint}")
    print(f"  model slug           : {inr_ckpt.get('model_name')}")
    print(f"  variant              : {inr_ckpt.get('variant', 'vanilla')}")
    print(f"  classifier type      : {getattr(classifier, '_clf_type', 'mlp')}")
    if getattr(inr, "is_spatial", False):
        print(
            "  spatial latent       : "
            f"{inr.latent_spatial_dim} x {inr.latent_spatial_dim} x {inr.latent_dim}"
        )
    print(f"  classifier accuracy  : {classifier_ckpt.get('accuracy')}")

    attack_model = FullPGDCIFAR10Spatial(
        inr,
        classifier,
        inner_steps=args.mod_steps,
        inner_lr=args.inner_lr,
        clean_grad_clip=args.clean_grad_clip,
        device=args.device,
    ).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    run_attack(
        attack_model,
        loader,
        criterion,
        args.epsilon / 255.0,
        args.pgd_steps,
        args.ext_lr,
        args.device,
        max_samples=args.max_samples,
        output_json=args.output_json,
        model_slug=inr_ckpt.get("model_name"),
        classifier_type=getattr(classifier, "_clf_type", None),
    )
