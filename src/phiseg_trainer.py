import argparse
import torchvision
import torch
import os
import shutil
import numpy as np
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from src.models.hierarchical_models import PHISeg
from src.metrics import per_label_dice
from src.datasets import *
from src.utils.data_transforms import ToTensor, convert_to_onehot, harden_softmax_outputs
from src.utils.io import save_segmentation


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_step(model, dataloader, optimizer, device, accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        x, y = batch['image'].to(device), batch['GT'].to(device)

        posterior_mus, posterior_sigmas, posterior_samples = model.posterior(x, y)
        prior_mus, prior_sigmas, _ = model.prior(x, posterior_samples=posterior_samples)
        y_hat = model.reconstruct_from_z(posterior_samples)

        kl_loss    = model.hierarchical_kl_loss(prior_mus, prior_sigmas, posterior_mus, posterior_sigmas)
        recon_loss = model.hierarchical_recon_loss(y_hat, y)
        loss = model.beta * kl_loss + recon_loss

        # scale down loss to average over accum_steps
        (loss / accum_steps).backward()
        total_loss += loss.item()

        # step and zero_grad every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()
        print(f'Batch {batch_idx+1}/{len(dataloader)}', end='\r')

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validation_step(model, dataloader, device, num_classes):
    model.eval()
    total_loss = 0
    dice_scores = []

    for batch in dataloader:
        x, y = batch['image'].to(device), batch['GT'].to(device)

        posterior_mus, posterior_sigmas, posterior_samples = model.posterior(x, y)
        prior_mus, prior_sigmas, _ = model.prior(x)
        y_hat = model.reconstruct_from_z(posterior_samples)

        kl_loss = model.hierarchical_kl_loss(prior_mus, prior_sigmas, posterior_mus, posterior_sigmas)
        recon_loss = model.hierarchical_recon_loss(y_hat, y)

        loss = model.beta * kl_loss + recon_loss
        total_loss += loss.item()

        y_pred_samples = model.predict_output_samples(x, N=5) # [B, N, C, H, W]
        y_pred = torch.mean(y_pred_samples, dim=1) # [B, C, H, W]
        y_pred = harden_softmax_outputs(y_pred, dim=1) # [B, C, H, W]

        y_onehot = convert_to_onehot(y, num_classes=num_classes)
        dice = per_label_dice(input=y_pred, target=y_onehot, input_is_batch=True)
        dice_scores.append(dice[1:].cpu().numpy())  # Exclude background class
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    dice_scores = np.stack(dice_scores)
    avg_dice = dice_scores.mean(axis=0)

    return avg_loss, avg_dice.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for PHISeg.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    set_seed(args.random_seed)

    checkpoint_dir = os.path.join("./phiseg_checkpoints", args.dataset)

    if args.dataset in ['psfhs', 'jsrt', 'wbc/cv', 'wbc/jtsc']:
        n_classes = 2
    else:
        n_classes = 1

    grayscale = True
    transforms_list = [ToTensor()]
    transforms = torchvision.transforms.Compose(transforms_list)
    target_size = 128 if '3d-ircadb' in args.dataset else 256

    if args.dataset == 'jsrt':
        transforms_list = [chestxray.Rescale(target_size), chestxray.ToTensorSeg(add_channel_dim=False)]
        transforms = torchvision.transforms.Compose(transforms_list)
        train_dataset, test_dataset, cal_dataset = chestxray.get_jsrt_datasets(transforms)
    else:
        train_dataset = TrainerDataset(split='Train', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size)
        cal_dataset = TrainerDataset(split='Calibration', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size)
        test_dataset = TrainerDataset(split='Test', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size)

    train_kwargs = {'pin_memory': True, 'batch_size': args.batch_size, 'shuffle': True}
    val_kwargs = {'pin_memory': True, 'batch_size': 1, 'shuffle': False}

    full_dataset = ConcatDataset([train_dataset, cal_dataset, test_dataset])
    train_loader = DataLoader(train_dataset, **train_kwargs, drop_last=True)
    cal_loader = DataLoader(cal_dataset, **val_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PHISeg(total_levels=7, latent_levels=5, zdim=2, num_classes=n_classes+1, beta=1.0).to(device)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(checkpoint_dir, exist_ok=True)

    effective_bs = args.batch_size * args.accum_steps
    print(f"Effective batch size: {effective_bs}")

    for epoch in tqdm(range(args.epochs), initial=1):
        train_loss = train_step(model, train_loader, optimizer, device, accum_steps=args.accum_steps)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nTrain loss: {train_loss:.2f}")

        val_loss, val_dice = validation_step(
            model=model,
            dataloader=cal_loader,
            device=device,
            num_classes=n_classes + 1,
        )
        val_dice_str = ", ".join(f"{d:.2f}" for d in val_dice)
        print(f"Val loss: {val_loss:.2f}, Val Dice per class: [{val_dice_str}]\n")

        base_name = f"epoch{epoch + 1}_bs{args.batch_size}_lr{args.lr:.0e}"
        checkpoint_name = f"{base_name}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
 
        counter = 1
        while os.path.exists(checkpoint_path):
            checkpoint_name = f"{base_name}_run{counter}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            counter += 1
 
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_name}")