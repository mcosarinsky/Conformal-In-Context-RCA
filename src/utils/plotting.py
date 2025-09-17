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