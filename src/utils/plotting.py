import seaborn as sns
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from .data_transforms import process_img

def select_k_random(dir, k=5):
    data_dir = dir / 'segs'
    labels_dir = dir / 'masks'
    seg_files = os.listdir(data_dir)
    unique_names = list(set([filename.split('_epoch')[0] for filename in seg_files]))
    n = len(unique_names)

    indices = np.random.permutation(n)[:k]

    to_tensor = transforms.ToTensor()
    selected_labels = []
    selected_segs = defaultdict(list)

    for idx in indices:
        img_name = unique_names[idx]
        # Handle the case where img_name contains underscores or not
        if 'irca' in img_name:
            label_path = os.path.join(labels_dir, img_name.replace('_', '/', 1) + '.png')

        elif '_' in img_name and 'ISIC' not in img_name:
            label_path = os.path.join(labels_dir, img_name.replace('_', '/').split('.')[0] + '.png')
        else:
            label_path = os.path.join(labels_dir, img_name + '.png')

        # Check if the label_path exists before processing
        if os.path.exists(label_path):
            label_tensor = to_tensor(process_img(label_path, 256, is_seg=True)) * 255
            selected_labels.append(label_tensor)
        else:
            print(f"Warning: File {label_path} not found.")
            selected_labels.append(None)

        for seg_path in seg_files:
            if img_name in seg_path:
                seg = Image.open(os.path.join(data_dir, seg_path)).convert('L')
                seg_tensor = to_tensor(seg) * 255

                epoch = seg_path.split('epoch')[1].split('.')[0]
                selected_segs[f'epoch_{epoch}'].append(seg_tensor)

    return selected_labels, selected_segs


def visualize_tensors(tensors, k, title=None):
    M = len(tensors)  # Number of keys (rows)
    cols = k  # Number of columns is the number of samples per key
    rows = M  # Each key corresponds to a row

    d = 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(d * cols, d * rows))

    if rows == 1:
        axes = axes.reshape(1, cols)
    if cols == 1:
        axes = axes.reshape(rows, 1)

    for row, (grp, tensor_list) in enumerate(tensors.items()):
        for col, tensor in enumerate(tensor_list[:k]):
            ax = axes[row, col]
            x = tensor.detach().cpu().numpy().squeeze()
            if len(x.shape) == 2:
                ax.imshow(x, vmin=x.min(), vmax=x.max())
            else:
                ax.imshow(np.transpose(x, (1, 2, 0)))

            if col == 0:
                ax.set_ylabel(grp, fontsize=16)

    # Hide any empty subplots
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.show()
    
def calculate_metrics(real_scores, pred_scores):
    """Calculate correlation and MAE for given scores, handling inf values."""
    real_scores = np.array(real_scores)
    pred_scores = np.array(pred_scores)
    
    # Filter out inf values
    valid = np.isfinite(real_scores) & np.isfinite(pred_scores)
    real_scores = real_scores[valid]
    pred_scores = pred_scores[valid]
    
    if len(real_scores) == 0:
        return np.nan, np.nan
    
    # Calculate correlation and MAE
    corr = np.corrcoef(real_scores, pred_scores)[0, 1]
    mae = np.mean(np.abs(real_scores - pred_scores))
    
    return corr, mae

def find_best_methods(eval_results):
    """
    Find which methods have the best correlation and MAE for each dataset.
    Returns dictionaries with bold_corr and bold_mae arrays for each dataset.
    """
    for eval_dict in eval_results:
        results = eval_dict['results']
        multiclass = eval_dict.get('is_multiclass', False)
        n_methods = len(results)
        
        if multiclass:
            # For multiclass, we need to handle each class separately
            n_classes = np.array(results[0]['Real']).shape[1] if len(results) > 0 else 0
            
            bold_corr = [[False] * n_classes for _ in range(n_methods)]
            bold_mae = [[False] * n_classes for _ in range(n_methods)]
            
            for class_idx in range(n_classes):
                # Calculate metrics for this class across all methods
                class_corrs = []
                class_maes = []
                
                for method_idx, res in enumerate(results):
                    real_class = np.array(res['Real'])[:, class_idx]
                    pred_class = np.array(res['Predicted'])[:, class_idx]
                    corr, mae = calculate_metrics(real_class, pred_class)
                    class_corrs.append(corr)
                    class_maes.append(mae)
                
                # Find best methods for this class (highest corr, lowest mae)
                # Handle ties by making all winners bold
                if not all(np.isnan(class_corrs)):
                    best_corr_value = np.nanmax(class_corrs)
                    # Round to 2 decimal places for tie detection
                    for idx, corr in enumerate(class_corrs):
                        if not np.isnan(corr) and round(corr, 2) == round(best_corr_value, 2):
                            bold_corr[idx][class_idx] = True
                
                if not all(np.isnan(class_maes)):
                    best_mae_value = np.nanmin(class_maes)
                    # Round to 2 decimal places for tie detection
                    for idx, mae in enumerate(class_maes):
                        if not np.isnan(mae) and round(mae, 2) == round(best_mae_value, 2):
                            bold_mae[idx][class_idx] = True
            
            eval_dict['bold_corr'] = bold_corr
            eval_dict['bold_mae'] = bold_mae
            
        else:
            # For single class
            bold_corr = [False] * n_methods
            bold_mae = [False] * n_methods
            
            # Calculate metrics for all methods
            method_corrs = []
            method_maes = []
            
            for res in results:
                corr, mae = calculate_metrics(res['Real'], res['Predicted'])
                method_corrs.append(corr)
                method_maes.append(mae)
            
            # Find best methods (highest corr, lowest mae)
            # Handle ties by making all winners bold
            if not all(np.isnan(method_corrs)):
                best_corr_value = np.nanmax(method_corrs)
                # Round to 2 decimal places for tie detection
                for idx, corr in enumerate(method_corrs):
                    if not np.isnan(corr) and round(corr, 2) == round(best_corr_value, 2):
                        bold_corr[idx] = True
            
            if not all(np.isnan(method_maes)):
                best_mae_value = np.nanmin(method_maes)
                # Round to 2 decimal places for tie detection
                for idx, mae in enumerate(method_maes):
                    if not np.isnan(mae) and round(mae, 2) == round(best_mae_value, 2):
                        bold_mae[idx] = True
            
            eval_dict['bold_corr'] = bold_corr
            eval_dict['bold_mae'] = bold_mae
    
    return eval_results

def plot_score(ax, real_scores, pred_scores, bold_corr=False, bold_mae=False):
    # Convert to numpy array
    real_scores = np.array(real_scores)
    pred_scores = np.array(pred_scores)

    # Filter out inf values
    valid = np.isfinite(real_scores) & np.isfinite(pred_scores)
    real_scores = real_scores[valid]
    pred_scores = pred_scores[valid]

    # Calculate correlation and MAE
    corr = np.corrcoef(real_scores, pred_scores)[0, 1]
    mae = np.mean(np.abs(real_scores - pred_scores))

    sns.scatterplot(x=real_scores, y=pred_scores, ax=ax)

    if bold_corr:
        corr_str = f"$\\mathbf{{Corr: {corr:.2f}}}$"
    else:
        corr_str = f"Corr: {corr:.2f}"

    if bold_mae:
        mae_str = f"$\\mathbf{{MAE: {mae:.2f}}}$"
    else:
        mae_str = f"MAE: {mae:.2f}"

    label_text = f"{corr_str}\n{mae_str}"

    ax.annotate(
        label_text,
        xy=(0.04, 0.87),  # Adjusted y-coordinate for annotation box
        xycoords='axes fraction',
        fontsize=13,  
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
    )

def plot_scores_multi(ax, real_scores, pred_scores, class_names=None, bold_corr_list=None, bold_mae_list=None):
    """
    Plot multi-class scores on the same axis, reporting MAE and correlation for each class.
    real_scores and pred_scores are expected to be 2D arrays with shape (n_samples, n_classes).
    """
    n_classes = real_scores.shape[1]  # Number of classes
    colors = sns.color_palette("colorblind", n_classes)  # Generate a color for each class

    if class_names is None:
        class_names = [f'Class {i + 1}' for i in range(n_classes)]

    if bold_corr_list is None:
        bold_corr_list = [False] * n_classes
    if bold_mae_list is None:
        bold_mae_list = [False] * n_classes

    for i in range(n_classes):
        real_class_scores = real_scores[:, i]
        pred_class_scores = pred_scores[:, i]

        # Filter out inf values
        valid = np.isfinite(real_class_scores) & np.isfinite(pred_class_scores)
        real_class_scores = real_class_scores[valid]
        pred_class_scores = pred_class_scores[valid]

        # Calculate correlation and MAE
        corr = np.corrcoef(real_class_scores, pred_class_scores)[0, 1]
        mae = np.mean(np.abs(real_class_scores - pred_class_scores))

        corr_str = f"Corr: {corr:.2f}"
        mae_str = f"MAE: {mae:.2f}"

        if bold_corr_list[i]:
            corr_str = f"$\\mathbf{{Corr: {corr:.2f}}}$"

        if bold_mae_list[i]:
            mae_str = f"$\\mathbf{{MAE: {mae:.2f}}}$"

        # Create the label text for the legend
        label_text = f"{class_names[i]} ({corr_str}, {mae_str})"

        # Plot the class scores with different colors
        sns.scatterplot(x=real_class_scores, y=pred_class_scores, ax=ax, color=colors[i],
                        label=label_text)

    ax.legend(loc=(0.01, 0.83), fancybox=True, fontsize=13, labelspacing=0.1, handletextpad=0.05)

def plot_results(eval_results, **kwargs):
    sns.set_theme(style="whitegrid")
    
    # Automatically determine which methods to bold
    eval_results = find_best_methods(eval_results)
    n_cols = kwargs.get('n_cols', 3)
    n_rows = len(eval_results)
    figsize = kwargs.get('figsize', (3 * n_cols, 3 * n_rows))

    fontsize = kwargs.get('fontsize', 17)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True,
                             figsize=figsize)
    axes = axes.flatten()

    for i, eval_dict in enumerate(eval_results):
        results = eval_dict['results']
        dataset = eval_dict['dataset']
        multiclass = eval_dict.get('is_multiclass', False)
        class_names = eval_dict.get('class_names', None)
        bold_corr = eval_dict.get('bold_corr', [False] * len(results))
        bold_mae = eval_dict.get('bold_mae', [False] * len(results))

        # Plot current row
        for j, res in enumerate(results):
            x = np.array(res['Real'])
            y = np.array(res['Predicted'])
            ax = axes[i * n_cols + j]  # Determine the correct axis for the current plot

            if multiclass:
                plot_scores_multi(ax, x, y, class_names=class_names,
                                  bold_corr_list=bold_corr[j], bold_mae_list=bold_mae[j])
            else:
                plot_score(ax, x, y, bold_corr=bold_corr[j], bold_mae=bold_mae[j])

            if j == 0:  # Set the dataset as ylabel of the first ax in current row
                ax.set_ylabel(dataset, fontsize=fontsize)

            if i == 0 and kwargs.get('titles') is not None:
                ax.set_title(kwargs['titles'][j], fontsize=fontsize)
            ax.set_box_aspect(1)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=-0.1)

    return fig

def plot_regressions(data_red, dataset_class_pairs):
    """
    Plots regression lines of predicted vs real values for different methods.
    """
    n_plots = len(dataset_class_pairs)
    methods = data_red.keys()
    method_labels = {
        'useg': 'UniverSeg',
        'sam': 'SAM',
        'atlas': 'Atlas',
        'atlas-ra': 'Atlas-RA'
    }
    method_colors = {
        'useg': '#1f77b4',       # Blue
        'sam': '#ff7f0e',        # Orange
        'atlas': '#2ca02c',      # Green
        'atlas-ra': '#d62728'    # Red
    }

    class_names_dict = {
    'jsrt': ['Lung', 'Heart'],
    'wbc_cv': ['Cyt', 'Nuc'],
    'wbc_jtsc': ['Cyt', 'Nuc'],
    }

    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
    axes = axes.flatten()

    for idx, (dataset_name, class_idx) in enumerate(dataset_class_pairs):
        ax = axes[idx]
        methods_data = {}

        for method in methods:
            res = data_red[method][dataset_name]['test']
            real = np.array(res['Real'])
            pred = np.array(res['Predicted'])

            if real.ndim == 1:
                real = np.expand_dims(real, axis=1)
                pred = np.expand_dims(pred, axis=1)

            real_class = real[:, class_idx]
            pred_class = pred[:, class_idx]
            valid = np.isfinite(real_class) & np.isfinite(pred_class)
            real_class = real_class[valid]
            pred_class = pred_class[valid]

            corr, mae = compute_metrics(real_class[:, np.newaxis], pred_class[:, np.newaxis])
            methods_data[method] = (corr[0], mae[0])

            df = pd.DataFrame({'Real': real_class, 'Predicted': pred_class})
            sns.regplot(
                x='Real', y='Predicted', data=df, ax=ax,
                label=method_labels[method],
                color=method_colors[method], scatter=False,
                line_kws={'linewidth': 3}, ci=95
            )

        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.7, label='Perfect estimation')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        if dataset_name in class_names_dict:
            class_name = class_names_dict[dataset_name][class_idx]
            title_text = f"{dataset_name.upper()} – {class_name}"
        else:
            # If there's only one class, omit class name
            has_multiple_classes = sum(d == dataset_name for d, _ in dataset_class_pairs) > 1
            if has_multiple_classes:
                title_text = f"{dataset_name.upper()} – Class {class_idx}"
            else:
                title_text = dataset_name.upper()

        ax.set_title(title_text, fontsize=15)
        ax.set_xlabel("Real DSC")
        ax.set_ylabel("Predicted DSC")

        bold_corr_methods, bold_mae_methods = find_best_metrics(methods_data)

        legend_elements = []
        for method in methods:
            if method in methods_data:
                corr, mae = methods_data[method]
                color = method_colors[method]
                corr_str = f"$\\mathbf{{Corr: {corr:.2f}}}$" if method in bold_corr_methods else f"Corr: {corr:.2f}"
                mae_str = f"$\\mathbf{{MAE: {mae:.2f}}}$" if method in bold_mae_methods else f"MAE: {mae:.2f}"
                legend_elements.append(
                    Line2D([0], [0], color=color, linewidth=3,
                           label=f"{method_labels[method]}: {corr_str}, {mae_str}")
                )

        legend = ax.legend(handles=legend_elements, loc='upper left',
                           bbox_to_anchor=(0.02, 0.98), fontsize=10,
                           frameon=True, fancybox=False, shadow=False,
                           facecolor='white', edgecolor='black', framealpha=1.0,
                           borderpad=0.3, handlelength=1.5, handletextpad=0.3,
                           columnspacing=0.5, labelspacing=0.2)
        legend.get_frame().set_boxstyle("round,pad=0.2")
        legend.get_frame().set_linewidth(1)

    # Hide unused subplots
    for idx in range(len(dataset_class_pairs), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    return fig

def compute_metrics(real_scores, pred_scores):
    """
    Compute correlation and MAE for each class separately.
    real_scores and pred_scores should be of shape (n_samples, n_classes).
    Returns two lists: correlations and MAEs per class.
    """
    n_classes = real_scores.shape[1]
    corr_list = []
    mae_list = []

    for i in range(n_classes):
        real = real_scores[:, i]
        pred = pred_scores[:, i]
        valid = np.isfinite(real) & np.isfinite(pred)
        real = real[valid]
        pred = pred[valid]

        if len(real) < 2:  # Not enough data to compute correlation
            corr = np.nan
        else:
            corr = np.corrcoef(real, pred)[0, 1]
        mae = np.mean(np.abs(real - pred))
        corr_list.append(corr)
        mae_list.append(mae)

    return corr_list, mae_list

def find_best_metrics(methods_data):
    """
    Find which methods have the best correlation and MAE.
    Returns sets of method names that should be bolded for corr and mae.
    """
    if not methods_data:
        return set(), set()
    
    # Extract correlations and MAEs
    correlations = []
    maes = []
    method_names = []
    
    for method, (corr, mae) in methods_data.items():
        if not (np.isnan(corr) and np.isnan(mae)):
            correlations.append(corr)
            maes.append(mae)
            method_names.append(method)
    
    if not correlations:
        return set(), set()
    
    # Find best values
    correlations = np.array(correlations)
    maes = np.array(maes)
    
    bold_corr_methods = set()
    bold_mae_methods = set()
    
    # Find best correlation (highest, handle ties)
    if not all(np.isnan(correlations)):
        best_corr = np.nanmax(correlations)
        for i, (method, corr) in enumerate(zip(method_names, correlations)):
            if not np.isnan(corr) and round(corr, 2) == round(best_corr, 2):
                bold_corr_methods.add(method)
    
    # Find best MAE (lowest, handle ties)
    if not all(np.isnan(maes)):
        best_mae = np.nanmin(maes)
        for i, (method, mae) in enumerate(zip(method_names, maes)):
            if not np.isnan(mae) and round(mae, 2) == round(best_mae, 2):
                bold_mae_methods.add(method)
    
    return bold_corr_methods, bold_mae_methods