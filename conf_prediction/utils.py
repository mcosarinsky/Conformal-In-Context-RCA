import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_stats(arrays, real_vals, estimator='mean', use_iqr=False, class_idx=None, n_samples=None, seed=1):
    # Fix seed for reproducibility
    rng = np.random.default_rng(seed)

    # Subsample
    if n_samples is not None:
        total = len(arrays)
        n = min(n_samples, total)
        indices = rng.choice(total, size=n, replace=False)
        arrays = [arrays[i] for i in indices]
        real_vals = [real_vals[i] for i in indices]

    # Convert elements and extract class_idx if needed
    processed = []
    for a in arrays:
        a = np.array(a)
        if class_idx is not None:
            a = a[:, class_idx]  # shape (n_i,)
        processed.append(a)

    if class_idx is not None:
        real_vals = [rv[class_idx] for rv in real_vals]

    # Compute statistics per sample
    central = []
    low = []
    high = []
    for a in processed:
        if estimator == 'mean':
            c = np.mean(a)
        elif estimator == 'median':
            c = np.median(a)
        elif estimator == 'max':
            c = np.max(a)
        else:
            raise ValueError(f"Unsupported estimator: {estimator}")
        central.append(c)

        if use_iqr:
            l = np.percentile(a, 25)
            h = np.percentile(a, 75)
        else:
            l = np.min(a)
            h = np.max(a)

        # Prevent negative error bars
        low.append(min(l, c))
        high.append(max(h, c))

    df = pd.DataFrame({
        'real': real_vals,
        'central': central,
        'low': low,
        'high': high
    }).sort_values(by='real').reset_index(drop=True)

    return df

# Plotting functions
def plot_rca_range_vs_real(
    dataset_results,
    dataset_name,
    estimator='mean',  # 'mean', 'median', 'max'
    use_iqr=False,
    n_samples=None,  # Optional: number of samples to subsample
    class_names=None  # Optional: list of class names for multiclass plots
):
    r = dataset_results[dataset_name]
    rca_arrays = r['RCA score']
    real_scores = r['Real score']

    # Detect multiclass
    is_multiclass = isinstance(real_scores[0], (list, np.ndarray))
    num_classes = len(real_scores[0]) if is_multiclass else 1

    for i in range(num_classes):
        class_label = (
            class_names[i] if class_names else (f"Class {i}" if is_multiclass else None)
        )

        df = compute_stats(
            rca_arrays,
            real_scores,
            estimator=estimator,
            use_iqr=use_iqr,
            class_idx=i if is_multiclass else None,
            n_samples=n_samples
        )

        plt.figure(figsize=(15, 5))
        yerr = [df['central'] - df['low'], df['high'] - df['central']]
        plt.errorbar(df.index, df['central'], yerr=yerr, fmt='o', color='C0',
                     ecolor='lightgray', capsize=3,
                     label=f'RCA {estimator.capitalize()} + Range', alpha=0.7)
        plt.plot(df.index, df['real'], 'o', color='C3', markersize=5, label='Real Score')

        title = f"{dataset_name.upper()} — RCA {estimator.capitalize()} vs Real Score"
        if class_label:
            title += f" — {class_label}"

        plt.title(title)
        plt.xlabel("Samples (sorted by Real Score)")
        plt.ylabel("Dice Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

def plot_interval_size(all_model_results, title=''):

    all_data = []

    for model, conformal_results in all_model_results.items():
        for method in conformal_results:
            for res in conformal_results[method]:
                intervals = np.array(res['intervals'])
                widths = intervals[:, 1] - intervals[:, 0]
                class_idx = res.get('class', 0)

                for w in widths:
                    all_data.append({
                        'Interval Width': w,
                        'Method': method,
                        'Model': model,
                        'Class': f'Class {class_idx}'
                    })

    df = pd.DataFrame(all_data)
    n_classes = len(df['Class'].unique())
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5), sharey=True)

    if n_classes == 1:
        axes = [axes]

    for i, (cls, group) in enumerate(df.groupby('Class')):
        ax = axes[i]
        # boxplot grouped by Method and colored by Model
        sns.boxplot(
            data=group,
            x='Method',
            y='Interval Width',
            hue='Model',
            ax=ax
        )
        ax.set_title(cls)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(bottom=0)
        ax.legend(title='Method', loc='upper left')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_coverage(conformal_results, alpha=0.1, title=''):
    """
    Plot empirical coverage for multiple conformal result sets.
    """
    all_data = []

    for method_label, results in conformal_results.items():
        for res in results:
            y_test = np.array(res['y_test'])
            intervals = np.array(res['intervals'])
            covered = np.logical_and(y_test >= intervals[:, 0], y_test <= intervals[:, 1])
            coverage = covered.mean()
            class_idx = res.get('class', 0)

            all_data.append({
                'Coverage': coverage,
                'Method': method_label,
                'Class': f'Class {class_idx}'
            })

    df = pd.DataFrame(all_data)
    n_classes = len(df['Class'].unique())
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5), sharey=True)

    if n_classes == 1:
        axes = [axes]

    for i, (cls, group) in enumerate(df.groupby('Class')):
        sns.barplot(data=group, x='Method', y='Coverage', ax=axes[i])
        axes[i].set_title(cls)
        axes[i].axhline(1 - alpha, color='red', linestyle='--', label='Target')
        axes[i].set_ylim(0, 1.05)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_coverage_analysis(conformal_results, alpha=0.1, conditioning='width', title=''):
    """
    Plot marginal and conditional coverage per method and class.
    
    Parameters:
    - conformal_results: dict of results per method.
    - alpha: desired error level.
    - conditioning: 'width' for interval width bins or 'value' for y_test value bins.
    - title: plot title.
    """
    assert conditioning in {'width', 'value'}, "conditioning must be 'width' or 'value'"
    
    all_data = []

    for method_label, results in conformal_results.items():
        for res in results:
            y_test = np.array(res['y_test'])
            intervals = np.array(res['intervals'])
            class_idx = res.get('class', 0)

            # Marginal
            covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
            all_data.append({
                'Coverage': covered.mean(),
                'Method': method_label,
                'Condition': 'Marginal',
                'Class': f'Class {class_idx}'
            })

            if conditioning == 'width':
                widths = intervals[:, 1] - intervals[:, 0]
                bin_labels = bin_by_width(widths)
            else:  
                bin_labels = bin_by_value(y_test)

            for label in np.unique(bin_labels):
                mask = bin_labels == label
                if np.any(mask):
                    covered_bin = (y_test[mask] >= intervals[mask, 0]) & (y_test[mask] <= intervals[mask, 1])
                    all_data.append({
                        'Coverage': covered_bin.mean(),
                        'Method': method_label,
                        'Condition': label,
                        'Class': f'Class {class_idx}'
                    })

    # Plotting
    df = pd.DataFrame(all_data)
    n_classes = len(df['Class'].unique())
    unique_labels = df['Condition'].unique()
    fig, axes = plt.subplots(1, n_classes, figsize=(8 * n_classes, 6), sharey=True)

    if n_classes == 1:
        axes = [axes]

    for i, (cls, group) in enumerate(df.groupby('Class')):
        sns.barplot(
            data=group,
            x='Method',
            y='Coverage',
            hue='Condition',
            palette=sns.color_palette("viridis", n_colors=len(unique_labels)),
            ax=axes[i]
        )
        axes[i].set_title(f'{cls} - Coverage', fontsize=14)
        axes[i].set_xlabel('')
        axes[i].axhline(1 - alpha, color='red', linestyle='--', label=f'1 - α = {1 - alpha:.2f}')
        axes[i].set_ylim(0, 1.05)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].set_ylabel('Coverage', fontsize=13)


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.86, 0.74), fontsize=13)
    for ax in axes:
        ax.get_legend().remove()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()


def plot_interval_size_by_value(conformal_results, title='Interval Width by Value Bins'):
    """
    Plot interval width distributions per method and class, both marginal and conditioned on y_test values.
    """
    all_data = []

    for method in conformal_results:
        for res in conformal_results[method]:
            intervals = np.array(res['intervals'])
            y_test = np.array(res['y_test'])
            widths = intervals[:, 1] - intervals[:, 0]
            class_idx = res.get('class', 0)

            value_bins = bin_by_value(y_test)

            for w, bin_label in zip(widths, value_bins):
                all_data.append({
                    'Interval Width': w,
                    'Method': method,
                    'Class': f'Class {class_idx}',
                    'Value Bin': bin_label
                })

            # Add marginal (unconditioned) distribution
            for w in widths:
                all_data.append({
                    'Interval Width': w,
                    'Method': method,
                    'Class': f'Class {class_idx}',
                    'Value Bin': 'Overall'
                })

    df = pd.DataFrame(all_data)
    n_classes = len(df['Class'].unique())
    width = 18 if n_classes > 1 else 7
    height = 8 if n_classes > 1 else 5.5
    fig, axes = plt.subplots(1, n_classes, figsize=(width, height), sharey=True)

    if n_classes == 1:
        axes = [axes]

    # Define consistent hue order and color palette
    hue_order = ['Overall', 'Bad (<0.6)', 'Average (0.6–0.8)', 'Good (>0.8)']

    for i, (cls, group) in enumerate(df.groupby('Class')):
        sns.boxplot(
            data=group,
            x='Method',
            y='Interval Width',
            hue='Value Bin',
            hue_order=hue_order,
            ax=axes[i]
        )
        axes[i].set_title(f'{cls} - Interval Width by Segmentation quality', fontsize=14)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].set_ylabel('Interval Width', fontsize=13)
        axes[i].set_ylim(bottom=0)

    # Legend outside to the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.84), fontsize=14)
    for ax in axes:
        ax.get_legend().remove()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()


def bin_by_width(widths):
    """Return labels for width-based bins."""
    bins = [
        (0, 0.1, 'Tiny (0, 0.1]'),
        (0.1, 0.2, 'Small (0.1, 0.2]'),
        (0.2, 0.5, 'Large (0.2, 0.5]'),
        (0.5, 1.0, 'Very Large (0.5, 1]')
    ]
    labels = np.full_like(widths, '', dtype=object)
    for low, high, label in bins:
        mask = (widths > low) & (widths <= high)
        labels[mask] = label
    return labels


def bin_by_value(y_values):
    """Return labels for value-based bins."""
    bins = [
        (0.0, 0.6, 'Bad (<0.6)'),
        (0.6, 0.8, 'Average (0.6–0.8)'),
        (0.8, 1.0, 'Good (>0.8)')
    ]
    labels = np.full_like(y_values, '', dtype=object)
    for low, high, label in bins:
        if low is None:
            mask = y_values < high
        elif high is None:
            mask = y_values > low
        else:
            mask = (y_values >= low) & (y_values <= high)
        labels[mask] = label
    return labels