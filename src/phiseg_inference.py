import argparse
import torch
import os
from pathlib import Path
from tqdm import tqdm
import torchvision
import numpy as np
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.hierarchical_models import PHISeg
from src.datasets import TrainerDataset
from src.utils.data_transforms import ToTensor
from src.utils.io import save_segmentation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def run_inference(model, dataloader, device, output_dir, current_epoch, num_samples=10):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    for batch in dataloader:
        x = batch['image'].to(device)
        img_name = Path(batch['name'][0]).stem

        y_samples = model.predict_output_samples(x, N=num_samples)  # [B, N, C, H, W]
        y_samples = y_samples.squeeze(0)  # [N, C, H, W]

        for i in range(num_samples):
            y_sample = y_samples[i]  # [C, H, W]
            y_pred = torch.argmax(y_sample, dim=0)  # [H, W]

            if torch.unique(y_pred).numel() > 1:
                file_name = f"{img_name}_epoch{current_epoch + 1}_sample{i + 1}.png"
                save_segmentation(y_pred.cpu(), file_name, output_dir)

            torch.cuda.empty_cache()


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = torchvision.transforms.Compose([ToTensor()])
    target_size = 128 if '3d-ircadb' in args.dataset else 256

    test_dataset = TrainerDataset(split='Test', dataset=args.dataset, transform=transforms, grayscale=True, target_size=target_size)
    cal_dataset = TrainerDataset(split='Calibration', dataset=args.dataset, transform=transforms, grayscale=True, target_size=target_size)

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
        model = PHISeg(total_levels=7, latent_levels=5, zdim=2, num_classes=n_classes + 1, beta=1.0).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Inference on Calibration
        cal_output_dir = os.path.join(output_dir, "Calibration", "segs")
        run_inference(model, cal_loader, device, cal_output_dir, current_epoch=i, num_samples=args.N)

        # Inference on Test
        test_output_dir = os.path.join(output_dir, "Test", "segs")
        run_inference(model, test_loader, device, test_output_dir, current_epoch=i, num_samples=args.N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHISeg Inference Script")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name used for training.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--N", type=int, default=5, help="Number of samples to generate per image.")

    args = parser.parse_args()
    main(args)
