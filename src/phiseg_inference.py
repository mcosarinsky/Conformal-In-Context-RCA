import argparse
import torch
import torch.nn.functional as F
import os
from pathlib import Path
from tqdm import tqdm
import json
import torchvision
import numpy as np
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.hierarchical_models import PHISeg
from src.datasets import *
from src.utils.data_transforms import ToTensor
from src.utils.io import save_segmentation
from src.metrics import predict_dice


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def run_inference(model, dataloader, device, output_dir, checkpoint_name, num_samples=10):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    results = {}  # Dictionary to store DSC stats per image

    for batch in dataloader:
        x = batch['image'].to(device)
        img_name = Path(batch['name'][0]).stem
        filename = f"{img_name}_{checkpoint_name}.png"

        y_samples = model.predict_output_samples(x, N=num_samples)  # [B, N, C, H, W]
        y_samples = y_samples.squeeze(0)  # [N, C, H, W]
        y_mean = torch.mean(y_samples, dim=0)  # [C, H, W]
        y_pred = torch.argmax(y_mean, dim=0)  # [H, W]

        if torch.unique(y_pred).numel() > 1:
            # Save predicted mask
            save_segmentation(y_pred.cpu(), filename, output_dir)

            # Compute soft DSCs per sample
            dsc_list = [predict_dice(F.softmax(y_samples[i], dim=0)) for i in range(num_samples)] 
            dsc_stack = torch.stack(dsc_list)  # [N, C-1]

            # Mean and std over samples for each class (excluding background)
            dsc_mean = dsc_stack.mean(dim=0).tolist()  
            dsc_std = dsc_stack.std(dim=0).tolist()    

            # Store results
            results[filename] = {
                "dsc_mean": dsc_mean,
                "dsc_std": dsc_std
            }

        torch.cuda.empty_cache()

    # Save results to JSON
    metrics_dir = os.path.join(output_dir, "predictions")
    os.makedirs(metrics_dir, exist_ok=True)
    json_path = os.path.join(metrics_dir, f"predicted_dsc_{checkpoint_name}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grayscale = True
    transforms = torchvision.transforms.Compose([ToTensor()])
    target_size = 128 if '3d-ircadb' in args.dataset else 256

    if args.dataset == 'jsrt':
        transforms_list = [chestxray.Rescale(target_size), chestxray.ToTensorSeg(add_channel_dim=False)]
        transforms = torchvision.transforms.Compose(transforms_list)
        train_dataset, test_dataset, cal_dataset = chestxray.get_jsrt_datasets(transforms)
    else:
        test_dataset = TrainerDataset(split='Test', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size)
        cal_dataset = TrainerDataset(split='Calibration', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size)

    val_kwargs = {'pin_memory': True, 'batch_size': 1, 'shuffle': False}
    test_loader = torch.utils.data.DataLoader(test_dataset, **val_kwargs)
    cal_loader = torch.utils.data.DataLoader(cal_dataset, **val_kwargs)

    if args.dataset in ['psfhs', 'jsrt', 'wbc/cv', 'wbc/jtsc']:
        n_classes = 2
    else:
        n_classes = 1

    checkpoints_dir = os.path.join("phiseg_checkpoints", args.dataset)
    checkpoint_paths = sorted(Path(checkpoints_dir).glob("*.pth"))
    output_dir = os.path.join("outputs", args.dataset)
    os.makedirs(output_dir + '/Calibration', exist_ok=True)
    os.makedirs(output_dir + '/Test', exist_ok=True)

    if not checkpoint_paths:
        print(f"No checkpoints found in {checkpoints_dir}")
        return

    for i, ckpt_path in enumerate(tqdm(checkpoint_paths, desc="Checkpoint")):
        ckpt_name = ckpt_path.stem
        model = PHISeg(total_levels=7, latent_levels=5, zdim=2, num_classes=n_classes + 1, beta=1.0).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        # Inference on Calibration
        cal_output_dir = os.path.join(output_dir, "Calibration")
        run_inference(model, cal_loader, device, cal_output_dir, ckpt_name, num_samples=args.N)

        # Inference on Test
        test_output_dir = os.path.join(output_dir, "Test")
        run_inference(model, test_loader, device, test_output_dir, ckpt_name, num_samples=args.N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHISeg Inference Script")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name used for training.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--N", type=int, default=50, help="Number of samples to generate per image.")

    args = parser.parse_args()
    main(args)
