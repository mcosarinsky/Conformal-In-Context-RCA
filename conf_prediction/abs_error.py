import numpy as np

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
    return (err / max(s, 0.025)) * scale_factor

def conformal_calibrate(dataset_dict, estimator='mean', alpha=0.1, sigma_type='std', scale_factor=1.0, n=50):
    """
    Perform conformal calibration for single or multi-class case.

    dataset_dict contains:
      - 'cal': {'Real score': array, 'RCA score': list of arrays}
      - 'test': {'Real score': array, 'RCA score': list of arrays}

    Returns:
        - For multiclass: list of dicts (one per class)
        - For single-class: single dict
    """
    real_cal = np.array(dataset_dict['cal']['Real score'])
    real_test = np.array(dataset_dict['test']['Real score'])
    rca_cal = np.array(dataset_dict['cal']['RCA score'])[:, :n]
    rca_test = np.array(dataset_dict['test']['RCA score'])[:, :n]

    is_mc = np.ndim(real_cal) == 2
    k = real_cal.shape[1] if is_mc else 1

    results = []

    for class_idx in range(k):
        y_cal = real_cal[:, class_idx] if is_mc else real_cal
        y_test = real_test[:, class_idx] if is_mc else real_test
        rca_cal_class = rca_cal[:, :, class_idx] if is_mc else rca_cal
        rca_test_class = rca_test[:, :, class_idx] if is_mc else rca_test

        # Calibration scores
        scores = [
            nonconformity_score(rca_cal_class[i], y_cal[i], estimator, sigma_type, scale_factor)
            for i in range(len(y_cal))
        ]
        N = len(scores)
        qhat = np.quantile(scores, np.ceil((N + 1) * (1 - alpha)) / N, method='higher')

        preds = []
        intervals = []
        test_scores = []

        for i in range(len(y_test)):
            arr = rca_test_class[i]
            y_hat = hat_y(arr, estimator)
            preds.append(y_hat)

            if sigma_type == 'abs':
                half_width = qhat
            elif sigma_type == 'std':
                s = np.std(arr, ddof=1)
                half_width = qhat * max(s, 0.025)
            elif sigma_type == 'iqr':
                s = sigma_iqr(arr)
                half_width = qhat * max(s, 0.025)
            elif sigma_type == 'trimmed':
                s = sigma_trimmed(arr)
                half_width = qhat * max(s, 0.025)
            elif sigma_type == 'mad':
                s = sigma_mad(arr)
                half_width = qhat * max(s, 0.025)
            else:
                raise ValueError(f"Unknown sigma_type: {sigma_type}")

            lo = max(0.0, y_hat - half_width)
            hi = min(1.0, y_hat + half_width)
            intervals.append((lo, hi))

            test_scores.append(nonconformity_score(arr, y_test[i], estimator, sigma_type, scale_factor))

        result = {
            'intervals': intervals,
            'y_test': y_test,
            'y_hat': preds,
            'q': qhat,
            'scores': scores,
            'test_scores': test_scores,
        }
        if is_mc:
            result['class'] = class_idx

        results.append(result)

    return results
