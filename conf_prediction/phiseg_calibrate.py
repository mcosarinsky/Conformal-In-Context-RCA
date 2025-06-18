from src.datasets import *
from src.metrics import compute_scores_by_name
import re
import os
import json
import numpy as np
import torchvision


def calibrate_dataset(dataset, alpha=0.1):
    target_size = 128 if '3d-ircadb' in dataset else 256
    
    if dataset in ['psfhs', 'jsrt', 'wbc/cv', 'wbc/jtsc']:
        n_classes = 2
    else:
        n_classes = 1

    if dataset == 'jsrt':
        transforms_list = [chestxray.Rescale(target_size), chestxray.ToTensorSeg(), chestxray.ToNumpy()] 
        transforms = torchvision.transforms.Compose(transforms_list)
        _, d_test, d_cal = chestxray.get_jsrt_datasets(transforms)
    else:
        d_test = Seg2D_Dataset(split='Test', dataset=dataset, target_size=target_size)
        d_cal = Seg2D_Dataset(split='Calibration', dataset=dataset, target_size=target_size)

    dsc_test = compute_scores_by_name(d_test, n_classes)
    dsc_cal = compute_scores_by_name(d_cal, n_classes)

    predicted_test = retrieve_dsc_predictions(d_test, f'outputs/{dataset}/Test/predictions')
    predicted_cal = retrieve_dsc_predictions(d_cal, f'outputs/{dataset}/Calibration/predictions')

    outputs = []

    # Align on shared keys
    common_names_cal = sorted(set(dsc_cal) & set(predicted_cal))
    common_names_test = sorted(set(dsc_test) & set(predicted_test))

    if len(common_names_cal) != len(d_cal):
        raise ValueError(f"Mismatch in calibration set: {len(common_names_cal)} predictions vs {len(d_cal)} samples.")

    if len(common_names_test) != len(d_test):
        raise ValueError(f"Mismatch in test set: {len(common_names_test)} predictions vs {len(d_test)} samples.")

    for i in range(n_classes):
        y_cal = np.array([dsc_cal[name][i] for name in common_names_cal])
        y_test = np.array([dsc_test[name][i] for name in common_names_test])
        pred_cal_mean = np.array([predicted_cal[name]['mean'][i] for name in common_names_cal])
        pred_cal_std = np.array([predicted_cal[name]['std'][i] for name in common_names_cal])
        pred_test_mean = np.array([predicted_test[name]['mean'][i] for name in common_names_test])
        pred_test_std = np.array([predicted_test[name]['std'][i] for name in common_names_test])

        scores = np.abs(y_cal - pred_cal_mean) / (pred_cal_std + 1e-6)
        N = len(scores)
        qhat = np.quantile(scores, np.ceil((N + 1) * (1 - alpha)) / N, method='higher')

        lower = np.clip(pred_test_mean - qhat * pred_test_std, 0.0, 1.0)
        upper = np.clip(pred_test_mean + qhat * pred_test_std, 0.0, 1.0)
        intervals = list(zip(lower, upper))

        outputs.append({
            'class': i,
            'y_test': y_test.tolist(),
            'y_hat': pred_test_mean.tolist(),
            'std': pred_test_std.tolist(),
            'scores': scores.tolist(),
            'q': qhat,
            'intervals': intervals
        })

    return outputs


def retrieve_dsc_predictions(dataset, path):
    dsc_preds = {}

    if 'jsrt' in path:
        path = path.replace('Test', 'F').replace('Calibration', 'M')
        regex = r'(epoch\d+)'
    else:
        regex = r'(check_\d+)'

    for sample in dataset:
        name = sample['seg_name']
        checkpoint = re.search(regex, name).group(1)
        json_file = f'{path}/predicted_dsc_{checkpoint}.json'

        if not os.path.exists(json_file):
            print(f"File {json_file} not found.")
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)
            if name not in data:
                print(f"{name} not found in {json_file}")
                continue
            mean, std = data[name]['dsc_mean'], data[name]['dsc_std']
            dsc_preds[name] = {'mean': mean, 'std': std}

    return dsc_preds