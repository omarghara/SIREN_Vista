import os
import os.path as osp
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloader import get_mnist_functa
from dataloader_modelnet import get_modelnet_functa
from utils import adjust_learning_rate, set_random_seeds, get_accuracy, Average


# -----------------------------------------------------------------------------
# Classifiers
# -----------------------------------------------------------------------------

class Classifier(nn.Module):
    """Flat MLP classifier for global or flattened Spatial Functa modulations."""

    def __init__(self, width=1024, depth=3, in_features=512, num_classes=10,
                 dropout=0.20, batchnorm=False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.net = self._make_layers(batchnorm=batchnorm)

    def _make_layers(self, batchnorm=False):
        num_features = [self.in_features] + [self.width] * self.depth + [self.num_classes]
        layers = []

        for i in range(self.depth):
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features[i]))
            layers.append(nn.Linear(num_features[i], num_features[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(num_features[self.depth], num_features[self.depth + 1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpatialPhiCNN(nn.Module):
    """
    CNN classifier for Spatial Functa modulations.

    Expected input to forward():
        (B, C, S, S)

    Your current Spatial Functa phi is saved as:
        (B, S, S, C), for example (B, 8, 8, 16)

    The conversion from (B, S, S, C) to (B, C, S, S) is done in
    _prepare_modulations(), not inside this class.
    """

    def __init__(self, in_channels=16, num_classes=10, width=128, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Dropout(dropout),
            nn.Linear(2 * width, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SpatialFunctaTransformerBlock(nn.Module):
    """
    Paper-style pre-LN Transformer block.

    Based on the Spatial Functa classifier description:
        LayerNorm -> Self-Attention -> Residual
        LayerNorm -> FFW -> Residual

    Dropout is intentionally NOT used inside the Transformer block because the
    paper states that dropout is applied once before the final linear layer.
    """

    def __init__(self, dim=128, hidden_dim=256, num_heads=16):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"Transformer dim={dim} must be divisible by num_heads={num_heads}."
            )

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffw = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # Pre-LN self-attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # Pre-LN feed-forward
        h = self.norm2(x)
        x = x + self.ffw(h)

        return x


class SpatialFunctaTransformer(nn.Module):
    """
    Paper-style Spatial Functa Transformer classifier.

    Expected input:
        (B, N, C)

    For your current setup:
        Spatial phi: (B, 8, 8, 16)
        Tokens:      (B, 64, 16)

    Architecture:
        phi tokens
        -> scalar normalization: phi / normalizing_factor
        -> Linear(C -> width)
        -> prepend learned CLS token
        -> add learned absolute positional embedding
        -> pre-LN Transformer blocks
        -> take CLS token
        -> LayerNorm
        -> Dropout
        -> Linear(width -> num_classes)

    Paper CIFAR-10 8x8x16 1-NN parameters:
        width = 128
        ffw_width = 256
        depth = 12
        heads = 16
        dropout = 0.1
        normalizing_factor = 0.08
    """

    def __init__(
        self,
        latent_dim=16,
        spatial_dim=8,
        num_classes=10,
        width=128,
        ffw_width=256,
        depth=12,
        num_heads=16,
        dropout=0.1,
        normalizing_factor=0.08,
    ):
        super().__init__()

        if width % num_heads != 0:
            raise ValueError(
                f"Transformer width {width} must be divisible by num_heads {num_heads}."
            )

        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        self.num_tokens = spatial_dim * spatial_dim
        self.num_classes = num_classes
        self.width = width
        self.ffw_width = ffw_width
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_prob = dropout
        self.normalizing_factor = normalizing_factor

        self.input_proj = nn.Linear(latent_dim, width)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens + 1, width))

        self.blocks = nn.ModuleList([
            SpatialFunctaTransformerBlock(
                dim=width,
                hidden_dim=ffw_width,
                num_heads=num_heads,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)

        self._init_weights()

    def _init_weights(self):
        # The unofficial Spatial Functa repo initializes cls/pos embeddings
        # with normal std=1.0, so we follow that style here.
        nn.init.normal_(self.cls_token, std=1.0)
        nn.init.normal_(self.pos_embedding, std=1.0)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape

        if N != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} spatial tokens, got {N}."
            )

        if C != self.latent_dim:
            raise ValueError(
                f"Expected token dim {self.latent_dim}, got {C}."
            )

        # Paper scalar normalization.
        x = x / self.normalizing_factor

        x = self.input_proj(x)  # (B, N, width)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, width)
        x = torch.cat([cls, x], dim=1)          # (B, N+1, width)

        x = x + self.pos_embedding[:, :N + 1]

        for block in self.blocks:
            x = block(x)

        cls_out = x[:, 0]
        cls_out = self.norm(cls_out)
        cls_out = self.dropout(cls_out)
        logits = self.head(cls_out)

        return logits


# -----------------------------------------------------------------------------
# Modulation preparation + normalization
# -----------------------------------------------------------------------------

def _prepare_modulations(phi, classifier_type, latent_spatial_dim=None, latent_dim=None):
    """
    Prepare Functa modulations for the selected classifier.

    MLP:
        global phi:  (B, D)       -> (B, D)
        spatial phi: (B, S, S, C) -> (B, S*S*C)

    CNN:
        spatial phi: (B, S, S, C) -> (B, C, S, S)
        flat phi:    (B, S*S*C)   -> reshape -> (B, C, S, S)

    ViT / Spatial Functa Transformer:
        spatial phi: (B, S, S, C) -> (B, S*S, C)
        flat phi:    (B, S*S*C)   -> reshape -> (B, S*S, C)
    """
    if classifier_type == "mlp":
        if phi.dim() > 2:
            phi = phi.reshape(phi.size(0), -1)
        return phi

    if classifier_type == "cnn":
        if phi.dim() == 4:
            # (B, S, S, C) -> (B, C, S, S)
            return phi.permute(0, 3, 1, 2).contiguous()

        if phi.dim() == 2:
            if latent_spatial_dim is None or latent_dim is None:
                raise ValueError(
                    "CNN classifier received flattened phi, but "
                    "--latent-spatial-dim and --latent-dim were not provided."
                )

            expected_dim = latent_spatial_dim * latent_spatial_dim * latent_dim
            if phi.size(1) != expected_dim:
                raise ValueError(
                    f"Cannot reshape phi of dim {phi.size(1)} into "
                    f"({latent_spatial_dim}, {latent_spatial_dim}, {latent_dim}); "
                    f"expected flat dim {expected_dim}."
                )

            phi = phi.view(
                phi.size(0),
                latent_spatial_dim,
                latent_spatial_dim,
                latent_dim,
            )
            return phi.permute(0, 3, 1, 2).contiguous()

        raise ValueError(f"CNN classifier expected phi with dim 2 or 4, got shape {tuple(phi.shape)}")

    if classifier_type == "vit":
        if phi.dim() == 4:
            # (B, S, S, C) -> (B, S*S, C)
            return phi.reshape(phi.size(0), -1, phi.size(-1)).contiguous()

        if phi.dim() == 2:
            if latent_spatial_dim is None or latent_dim is None:
                raise ValueError(
                    "ViT classifier received flattened phi, but "
                    "--latent-spatial-dim and --latent-dim were not provided."
                )

            expected_dim = latent_spatial_dim * latent_spatial_dim * latent_dim
            if phi.size(1) != expected_dim:
                raise ValueError(
                    f"Cannot reshape phi of dim {phi.size(1)} into "
                    f"({latent_spatial_dim * latent_spatial_dim}, {latent_dim}); "
                    f"expected flat dim {expected_dim}."
                )

            return phi.view(
                phi.size(0),
                latent_spatial_dim * latent_spatial_dim,
                latent_dim,
            ).contiguous()

        raise ValueError(f"ViT classifier expected phi with dim 2 or 4, got shape {tuple(phi.shape)}")

    raise ValueError(f"Unknown classifier_type: {classifier_type}")


@torch.no_grad()
def compute_phi_normalization_stats(loader, classifier_type, latent_spatial_dim, latent_dim, device):
    """
    Compute normalization statistics from the training functaset only.

    For MLP:
        mean/std per flattened phi dimension: (1, D)

    For CNN:
        mean/std per latent channel: (1, C, 1, 1)

    For ViT:
        mean/std per latent channel over all tokens: (1, 1, C)

    Note:
        For the first paper-style ViT run, do NOT pass --normalize-phi.
        The paper-style scalar normalization is already inside
        SpatialFunctaTransformer: phi / normalizing_factor.
    """
    all_phi = []

    for phi, _ in tqdm(loader, desc="[norm] computing phi stats"):
        phi = phi.to(device, non_blocking=True).float()
        phi = _prepare_modulations(
            phi,
            classifier_type=classifier_type,
            latent_spatial_dim=latent_spatial_dim,
            latent_dim=latent_dim,
        )
        all_phi.append(phi.detach().cpu())

    all_phi = torch.cat(all_phi, dim=0)

    if classifier_type == "mlp":
        mean = all_phi.mean(dim=0, keepdim=True)
        std = all_phi.std(dim=0, keepdim=True)
    elif classifier_type == "cnn":
        # all_phi shape: (B, C, S, S)
        mean = all_phi.mean(dim=(0, 2, 3), keepdim=True)
        std = all_phi.std(dim=(0, 2, 3), keepdim=True)
    elif classifier_type == "vit":
        # all_phi shape: (B, N, C)
        mean = all_phi.mean(dim=(0, 1), keepdim=True)
        std = all_phi.std(dim=(0, 1), keepdim=True)
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    std = std.clamp_min(1e-6)
    return mean.to(device), std.to(device)


def apply_phi_normalization(phi, norm_stats):
    if norm_stats is None:
        return phi
    mean, std = norm_stats
    return (phi - mean) / std


# -----------------------------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------------------------

def train_classifier(model, train_loader, optimizer, criterion, epoch,
                     classifier_type="mlp", latent_spatial_dim=None,
                     latent_dim=None, norm_stats=None,
                     max_steps_this_epoch=None):
    model.train()
    device = next(iter(model.parameters())).device
    losses = []
    train_score = 0
    samples_seen = 0
    steps_done = 0

    prog_bar = tqdm(train_loader, total=len(train_loader))
    for images, labels in prog_bar:
        if max_steps_this_epoch is not None and steps_done >= max_steps_this_epoch:
            break

        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        images = _prepare_modulations(
            images,
            classifier_type=classifier_type,
            latent_spatial_dim=latent_spatial_dim,
            latent_dim=latent_dim,
        )
        images = apply_phi_normalization(images, norm_stats)

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        train_score += preds.argmax(dim=-1).eq(labels).sum().item()
        samples_seen += labels.size(0)
        steps_done += 1

    accuracy = train_score / max(samples_seen, 1)
    mean_loss = sum(losses) / max(len(losses), 1)

    print('epoch: %d, loss: %.4f, train acc: %.3f%s' % (
        epoch, mean_loss, accuracy * 100, '%'
    ))

    return losses, steps_done


@torch.no_grad()
def eval_classifier(model, val_loader, epoch, classifier_type="mlp",
                    latent_spatial_dim=None, latent_dim=None, norm_stats=None):
    model.eval()
    device = next(iter(model.parameters())).device

    prog_bar = tqdm(val_loader, total=len(val_loader))
    top1acc = Average()
    top5acc = Average()

    for images, labels in prog_bar:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        images = _prepare_modulations(
            images,
            classifier_type=classifier_type,
            latent_spatial_dim=latent_spatial_dim,
            latent_dim=latent_dim,
        )
        images = apply_phi_normalization(images, norm_stats)

        preds = model(images)
        top1acc_batch, top5acc_batch = get_accuracy(preds, labels, top_k=(1, 5))
        top1acc.update(top1acc_batch, labels.size(0))
        top5acc.update(top5acc_batch, labels.size(0))

    print('epoch: %d, val accuracy: top1 %.2f%s, top5 %.2f%s' % (
        epoch, top1acc.avg, '%', top5acc.avg, '%'
    ))
    return top1acc.avg, top5acc.avg


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--lr', type=float, default=0.001, help='classifier optimization lr')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='optimizer weight decay')

    parser.add_argument(
        '--classifier-type',
        choices=['mlp', 'cnn', 'vit'],
        default='mlp',
        help='Downstream classifier: flat MLP, CNN over spatial phi, or paper-style Spatial Functa Transformer.'
    )

    # MLP args
    parser.add_argument('--cwidth', type=int, default=512, help='classifier MLP hidden dimension')
    parser.add_argument('--cdepth', type=int, default=3, help='classifier MLP depth')
    parser.add_argument('--mod-dim', type=int, default=512, help='flat modulation dimension')

    # CNN args
    parser.add_argument('--cnn-width', type=int, default=128, help='base channel width for CNN classifier')

    # Spatial latent shape args
    parser.add_argument(
        '--latent-spatial-dim',
        type=int,
        default=8,
        help='spatial side S of phi grid: phi shape (S, S, C)'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=16,
        help='channel dimension C of spatial phi grid'
    )

    # Paper-style ViT / Spatial Functa Transformer args
    parser.add_argument(
        '--vit-width',
        type=int,
        default=128,
        help='Transformer width. Paper CIFAR-10 8x8x16 1-NN uses 128.'
    )
    parser.add_argument(
        '--vit-ffw-width',
        type=int,
        default=256,
        help='Transformer feed-forward width. Paper uses 2x width = 256.'
    )
    parser.add_argument(
        '--vit-depth',
        type=int,
        default=12,
        help='Number of Transformer blocks. Paper CIFAR-10 8x8x16 1-NN uses 12.'
    )
    parser.add_argument(
        '--vit-heads',
        type=int,
        default=16,
        help='Number of attention heads. Paper uses 16.'
    )
    parser.add_argument(
        '--vit-normalizing-factor',
        type=float,
        default=0.08,
        help='Scalar latent normalization factor. Paper CIFAR-10 8x8x16 1-NN uses 0.08.'
    )

    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate in classifier')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='label smoothing for CrossEntropyLoss')
    parser.add_argument('--normalize-phi', action='store_true', default=False,
                        help='normalize phi using training-set mean/std before classifier')

    parser.add_argument('--batch-size', type=int, default=256, help='optimization mini-batch size')
    parser.add_argument('--dataset', choices=["mnist", "fmnist", "cifar10", "modelnet"], required=True,
                        help="Train for MNIST, Fashion-MNIST, CIFAR-10, or ModelNet10")
    parser.add_argument('--num-epochs', type=int, default=160, help='number of classifier training epochs')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=None,
        help='Optional step-based training budget. Paper uses 100000 steps for CIFAR-10.'
    )

    parser.add_argument('--data-path', type=str, default='..', help='unused here; kept for compatibility')
    parser.add_argument('--functaset-path-train', type=str, required=True,
                        help='path to optimized training functaset pkl')
    parser.add_argument('--functaset-path-test', type=str, required=True,
                        help='path to optimized test functaset pkl')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Pass "cuda" to use GPU')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='directory to save classifier checkpoint and curves')

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()
    set_random_seeds(args.seed, args.device)

    loader_fn = get_modelnet_functa if args.dataset == "modelnet" else get_mnist_functa

    train_functaloader = loader_fn(
        data_dir=args.functaset_path_train,
        mode='train',
        batch_size=args.batch_size,
    )
    test_functaloader = loader_fn(
        data_dir=args.functaset_path_test,
        mode='test',
        batch_size=args.batch_size,
    )

    if args.classifier_type == "mlp":
        classifier = Classifier(
            width=args.cwidth,
            depth=args.cdepth,
            in_features=args.mod_dim,
            num_classes=10,
            dropout=args.dropout,
            batchnorm=args.dataset == "modelnet",
        ).to(args.device)

    elif args.classifier_type == "cnn":
        classifier = SpatialPhiCNN(
            in_channels=args.latent_dim,
            num_classes=10,
            width=args.cnn_width,
            dropout=args.dropout,
        ).to(args.device)

    elif args.classifier_type == "vit":
        classifier = SpatialFunctaTransformer(
            latent_dim=args.latent_dim,
            spatial_dim=args.latent_spatial_dim,
            num_classes=10,
            width=args.vit_width,
            ffw_width=args.vit_ffw_width,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            dropout=args.dropout,
            normalizing_factor=args.vit_normalizing_factor,
        ).to(args.device)

    else:
        raise ValueError(f"Unknown classifier type: {args.classifier_type}")

    print(f"[classifier] type: {args.classifier_type}")
    print(f"[classifier] model: {classifier.__class__.__name__}")
    print(f"[classifier] normalize_phi: {args.normalize_phi}")
    print(f"[classifier] num parameters: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")

    if args.classifier_type == "vit":
        print("[classifier] paper-style ViT scalar normalization:")
        print(f"  phi / {args.vit_normalizing_factor}")
        print("[classifier] recommendation: for first paper-style run, do NOT use --normalize-phi.")

    norm_stats = None
    if args.normalize_phi:
        norm_stats = compute_phi_normalization_stats(
            train_functaloader,
            classifier_type=args.classifier_type,
            latent_spatial_dim=args.latent_spatial_dim,
            latent_dim=args.latent_dim,
            device=args.device,
        )
        print("[classifier] computed phi mean/std normalization stats")

    if args.classifier_type == "vit":
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
    ).to(args.device)

    train_losses = []
    val_top1accs = []
    val_top5accs = []
    best_accuracy = 0.0
    best_epoch = -1
    global_step = 0

    if args.save_dir is not None:
        model_dir = args.save_dir
    else:
        if args.classifier_type == "mlp":
            model_dir = f"{args.dataset}_classifier"
        else:
            model_dir = f"{args.dataset}_{args.classifier_type}_classifier"

    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        if args.num_steps is not None and global_step >= args.num_steps:
            print(f"[classifier] reached num_steps={args.num_steps}; stopping.")
            break

        # Keep the original epoch LR schedule for MLP/CNN.
        # For paper-style ViT, we usually keep constant LR, so we do not call
        # adjust_learning_rate for classifier_type == "vit".
        if args.classifier_type != "vit":
            if args.dataset != "modelnet":
                adjust_learning_rate(optimizer, epoch, args.lr, args.num_epochs)
            elif epoch >= 100 and epoch % 10 == 0:
                optimizer.param_groups[0]['lr'] /= 2

        max_steps_this_epoch = None
        if args.num_steps is not None:
            max_steps_this_epoch = args.num_steps - global_step

        losses_epo, steps_done = train_classifier(
            model=classifier,
            train_loader=train_functaloader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            classifier_type=args.classifier_type,
            latent_spatial_dim=args.latent_spatial_dim,
            latent_dim=args.latent_dim,
            norm_stats=norm_stats,
            max_steps_this_epoch=max_steps_this_epoch,
        )
        global_step += steps_done
        train_losses.extend(losses_epo)

        top1acc, top5acc = eval_classifier(
            model=classifier,
            val_loader=test_functaloader,
            epoch=epoch,
            classifier_type=args.classifier_type,
            latent_spatial_dim=args.latent_spatial_dim,
            latent_dim=args.latent_dim,
            norm_stats=norm_stats,
        )
        val_top1accs.append(top1acc)
        val_top5accs.append(top5acc)

        if best_accuracy < top1acc:
            best_accuracy = top1acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': classifier.state_dict(),
                'accuracy': best_accuracy,
                'classifier_type': args.classifier_type,
                'dataset': args.dataset,
                'mod_dim': args.mod_dim,
                'latent_spatial_dim': args.latent_spatial_dim,
                'latent_dim': args.latent_dim,

                # CNN
                'cnn_width': args.cnn_width,

                # MLP
                'cwidth': args.cwidth,
                'cdepth': args.cdepth,

                # ViT / Spatial Functa Transformer
                'vit_width': args.vit_width,
                'vit_ffw_width': args.vit_ffw_width,
                'vit_depth': args.vit_depth,
                'vit_heads': args.vit_heads,
                'vit_normalizing_factor': args.vit_normalizing_factor,

                # Training
                'dropout': args.dropout,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'label_smoothing': args.label_smoothing,
                'normalize_phi': args.normalize_phi,
                'num_steps': args.num_steps,

                # Optional mean/std normalization
                'phi_mean': norm_stats[0].detach().cpu() if norm_stats is not None else None,
                'phi_std': norm_stats[1].detach().cpu() if norm_stats is not None else None,
            }, osp.join(model_dir, 'best_classifier.pth'))

    print(f"Best Accuracy: {best_accuracy:.2f} % at epoch {best_epoch}")
    print(f"Final global step: {global_step}")

    np.save(osp.join(model_dir, 'classifier_loss.npy'), np.array(train_losses))
    np.save(osp.join(model_dir, 'classifier_acc.npy'), np.array((val_top1accs, val_top5accs)))