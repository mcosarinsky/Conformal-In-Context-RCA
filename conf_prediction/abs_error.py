import numpy as np
from .utils import split_calibration_test

def hat_y(arr, estimator):
    """Central estimator: 'mean' or 'max'."""
    if estimator == 'mean':
        return np.mean(arr)
    elif estimator == 'max':
        return np.max(arr)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

def sigma_iqr(arr):
    """IQR‐based sigma"""
    q1, q3 = np.percentile(arr, [25, 75])
    return (q3 - q1) / 1.349

def sigma_trimmed(arr, trim_frac=0.1):
    """
    Trim the bottom `trim_frac` fraction of values before computing std.
    """
    lower = np.percentile(arr, 100 * trim_frac)
    filtered = arr[(arr >= lower)]
    return np.std(filtered, ddof=1) if len(filtered) > 1 else np.std(arr, ddof=1)

def sigma_mad(arr):
    """MAD-based sigma estimator (robust to outliers)."""
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return mad / 0.6745


def nonconformity_score(arr, y, estimator='mean', sigma_type='std', scale_factor=1.0):
    """
    Conformal nonconformity score S(x,y):
      - 'std':  |y - ŷ| / σ
      - 'iqr':  |y - ŷ| / (IQR/1.349)
      - 'trimmed': |y - ŷ| / σ_trimmed
      - 'mad':  |y - ŷ| / (MAD/0.6745)
      - 'abs':  |y - ŷ| 
    """
    y_hat = hat_y(arr, estimator)
    err = abs(y - y_hat)

    # Compute sigma based on the chosen type
    if sigma_type == 'std':
        s = np.std(arr, ddof=1)
    elif sigma_type == 'iqr':
        s = sigma_iqr(arr)
    elif sigma_type == 'trimmed':
        s = sigma_trimmed(arr, trim_frac=0.1)
    elif sigma_type == 'mad':
        s = sigma_mad(arr)
    elif sigma_type == 'abs':
        return err * scale_factor # For 'abs', no need for sigma
    else:
        raise ValueError(f"Unknown sigma_type: {sigma_type}")
    return (err / max(s, 1e-6)) * scale_factor

def conformal_calibrate(
    dataset_dict,
    estimator='mean',
    alpha=0.1,
    frac_cal=0.5,
    sigma_type='std',
    scale_factor=1.0,
    seed=42
):
    """
    Perform conformal calibration for single or multi-class case.
    dataset_dict: {'Real score': [...], 'RCA score': [...]}
    Returns:
        - For binary/multiclass: list of dicts (one per class)
        - For single-class: single dict
    """
    real = np.array(dataset_dict['Real score'])
    rca  = dataset_dict['RCA score']
    n = len(real)

    # Check if multiclass (k = number of classes)
    is_multiclass = np.ndim(real) == 2
    k = real.shape[1] if is_multiclass else 1

    # Split indices
    calib_idx, test_idx = split_calibration_test(n, frac_cal, seed)

    results = []

    for class_idx in range(k):
        # Extract class-specific values
        real_class = real[:, class_idx] if is_multiclass else real
        rca_class = [np.array(r)[:, class_idx] if is_multiclass else np.array(r) for r in rca]

        # Calibration scores
        scores = [
            nonconformity_score(rca_class[i], real_class[i], estimator, sigma_type, scale_factor)
            for i in calib_idx
        ]
        N = len(calib_idx)
        qhat = np.quantile(scores, np.ceil((N+1)*(1-alpha))/N, method='higher')

        # Prediction intervals
        preds = []
        intervals = []
        for i in test_idx:
            arr = np.array(rca_class[i])
            y_hat = hat_y(arr, estimator)
            preds.append(y_hat)

            if sigma_type == 'abs':
                half_width = qhat
            elif sigma_type == 'std':
                s = np.std(arr, ddof=1)
                half_width = qhat * max(s, 1e-6)
            elif sigma_type == 'iqr':
                s = sigma_iqr(arr)
                half_width = qhat * max(s, 1e-6)
            elif sigma_type == 'trimmed':
                s = sigma_trimmed(arr)
                half_width = qhat * max(s, 1e-6)
            elif sigma_type == 'mad':
                s = sigma_mad(arr)
                half_width = qhat * max(s, 1e-6)
            else:
                raise ValueError(f"Unknown sigma_type: {sigma_type}")

            lo = max(0.0, y_hat - half_width)
            hi = min(1.0, y_hat + half_width)
            intervals.append((lo, hi))

        result = {
            'intervals': intervals,
            'y_test': real_class[test_idx],
            'y_hat': preds,
            'q': qhat,
            'scores': scores,
        }

        if is_multiclass:
            result['class'] = class_idx
        results.append(result)
    return results 