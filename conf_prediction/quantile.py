import numpy as np
from .utils import split_calibration_test

def quantile_band_asymm(arr, p_l, p_h):
    """Asymmetric raw band: [p_l, p_h] quantiles of arr."""
    lo, hi = np.percentile(arr, [p_l * 100, p_h * 100])
    return lo, hi


def quantile_calibrate(
    dataset_dict,
    p_l=0.40, p_h=0.95,
    alpha=0.10,         
    frac_cal=0.5,
    seed=42
):
    """
    Standard (symmetric) CQR (Theorem 1):
    - alpha: total miscoverage budget.
    - Uses a single conformalization on combined nonconformity scores.
    """
    real = np.array(dataset_dict['Real score'])
    rca  = dataset_dict['RCA score']
    n    = len(real)

    is_mc = np.ndim(real) == 2
    k     = real.shape[1] if is_mc else 1

    calib_idx, test_idx = split_calibration_test(n, frac_cal, seed)

    outputs = []
    for class_idx in range(k):
        # extract class-specific data
        if is_mc:
            y_all   = real[:, class_idx]
            rca_all = [np.array(r)[:, class_idx] for r in rca]
        else:
            y_all   = real
            rca_all = rca

        # compute combined nonconformity scores on calibration
        S_cal = []
        for i in calib_idx:
            lo, hi = quantile_band_asymm(rca_all[i], p_l, p_h)
            y = y_all[i]
            S_cal.append(max(lo - y, y - hi))

        # inflation margin
        N = len(calib_idx)
        qhat = np.quantile(S_cal, np.ceil((N+1)*(1-alpha))/N, method='higher')

        # build final intervals on test set
        intervals, preds = [], []
        for i in test_idx:
            lo, hi = quantile_band_asymm(rca_all[i], p_l, p_h)
            lb = max(0, lo - qhat)
            ub = min(1, hi + qhat)
            intervals.append((lb, ub))
            preds.append(np.mean(rca_all[i]))

        outputs.append({
            'class': class_idx,
            'intervals': intervals,
            'y_test': y_all[test_idx],
            'y_hat':  preds,
        })

    return outputs


def quantile_calibrate_asymm(
    dataset_dict,
    p_l=0.40, p_h=0.95,
    alpha_l=0.09,      
    alpha_u=0.01,      
    frac_cal=0.5,
    seed=42
):
    """
    Asymmetric CQR (Theorem 2):
    - alpha: total miscoverage budget.
    - alpha_l, alpha_u: *pre-specified* tail budgets summing to alpha.
    """
    real = np.array(dataset_dict['Real score'])
    rca  = dataset_dict['RCA score']
    n    = len(real)

    is_mc = np.ndim(real) == 2
    k     = real.shape[1] if is_mc else 1

    calib_idx, test_idx = split_calibration_test(n, frac_cal, seed)

    outputs = []
    for class_idx in range(k):
        # --- extract class-specific arrays ---
        if is_mc:
            y_all   = real[:, class_idx]
            rca_all = [np.array(r)[:, class_idx] for r in rca]
        else:
            y_all   = real
            rca_all = rca

        # --- compute tail nonconformity scores on calibration ---
        S_l_cal, S_u_cal = [], []
        for i in calib_idx:
            lo, hi = quantile_band_asymm(rca_all[i], p_l, p_h)
            y = y_all[i]
            S_l_cal.append(max(lo - y, 0))
            S_u_cal.append(max(y - hi, 0))

        # inflation margins 
        N = len(calib_idx)
        q_l_quantile = min(np.ceil((N + 1) * (1 - alpha_l)) / N, 1.0)
        q_u_quantile = min(np.ceil((N + 1) * (1 - alpha_u)) / N, 1.0)
        q_l = np.quantile(S_l_cal, q_l_quantile, method='higher')
        q_u = np.quantile(S_u_cal, q_u_quantile, method='higher')
        #q_l = np.quantile(S_l_cal, np.ceil((N+1)*(1-alpha_l))/N, method='higher')
        #q_u = np.quantile(S_u_cal, np.ceil((N+1)*(1-alpha_u))/N, method='higher')

        # --- build final intervals on test set ---
        intervals, preds = [], []
        for i in test_idx:
            lo, hi = quantile_band_asymm(rca_all[i], p_l, p_h)
            lb = max(0, lo - q_l)
            ub = min(1, hi + q_u)
            intervals.append((lb, ub))
            preds.append(np.mean(rca_all[i]))

        result = {
            'class': class_idx,
            'intervals': intervals,
            'y_test':   y_all[test_idx],
            'y_hat':    preds,
        }
        outputs.append(result)

    return outputs


def quantile_calibrate_adaptative(
    dataset_dict,
    p_l=0.40, p_h=0.95,
    alpha=0.10,  
    frac_cal=0.5,
    seed=42
):
    """
    Standard (symmetric) CQR (Theorem 1):
    - alpha: total miscoverage budget.
    - Uses a single conformalization on combined nonconformity scores.
    """
    real = np.array(dataset_dict['Real score'])
    rca  = dataset_dict['RCA score']
    n    = len(real)

    is_mc = np.ndim(real) == 2
    k     = real.shape[1] if is_mc else 1

    calib_idx, test_idx = split_calibration_test(n, frac_cal, seed)

    outputs = []
    for class_idx in range(k):
        # extract class-specific data
        if is_mc:
            y_all   = real[:, class_idx]
            rca_all = [np.array(r)[:, class_idx] for r in rca]
        else:
            y_all   = real
            rca_all = rca

        # compute combined nonconformity scores on calibration
        S_cal = []
        for i in calib_idx:
            lo, hi = quantile_band_asymm(rca_all[i], p_l, p_h)
            y = y_all[i]
            scale = hi - lo + 1e-3

            S_cal.append(max(lo - y, y - hi)/scale)

        # inflation margin
        N = len(calib_idx)
        qhat = np.quantile(S_cal, np.ceil((N+1)*(1-alpha))/N, method='higher')

        # build final intervals on test set
        intervals, preds = [], []
        for i in test_idx:
            lo, hi = quantile_band_asymm(rca_all[i], p_l, p_h)
            scale = hi - lo + 1e-3
            lb = max(0, lo - qhat*scale)
            ub = min(1, hi + qhat*scale)
            intervals.append((lb, ub))
            preds.append(np.mean(rca_all[i]))

        outputs.append({
            'class': class_idx,
            'intervals': intervals,
            'y_test': y_all[test_idx],
            'y_hat':  preds,
        })

    return outputs


def quantile_calibrate_adaptative_asymm(
    dataset_dict,
    p_l=0.40, p_h=0.95,
    alpha_l=0.09,      
    alpha_u=0.01,      
    frac_cal=0.5,
    seed=42
):
    """
    Asymmetric CQR (Theorem 2):
    - alpha: total miscoverage budget.
    - alpha_l, alpha_u: *pre-specified* tail budgets summing to alpha.
    """
    real = np.array(dataset_dict['Real score'])
    rca  = dataset_dict['RCA score']
    n    = len(real)

    is_mc = np.ndim(real) == 2
    k     = real.shape[1] if is_mc else 1

    calib_idx, test_idx = split_calibration_test(n, frac_cal, seed)

    outputs = []
    for class_idx in range(k):
        # --- extract class-specific arrays ---
        if is_mc:
            y_all   = real[:, class_idx]
            rca_all = [np.array(r)[:, class_idx] for r in rca]
        else:
            y_all   = real
            rca_all = rca

        # --- compute tail nonconformity scores on calibration ---
        S_l_cal, S_u_cal = [], []
        for i in calib_idx:
            lo, hi = quantile_band_asymm(rca_all[i], p_l, p_h)
            y = y_all[i]
            scale = hi - lo + 1e-3
            S_l_cal.append(max(lo - y, 0) / scale)
            S_u_cal.append(max(y - hi, 0) / scale) 

        # inflation margins 
        N = len(calib_idx)
        q_l_quantile = min(np.ceil((N + 1) * (1 - alpha_l)) / N, 1.0)
        q_u_quantile = min(np.ceil((N + 1) * (1 - alpha_u)) / N, 1.0)
        q_l = np.quantile(S_l_cal, q_l_quantile, method='higher')
        q_u = np.quantile(S_u_cal, q_u_quantile, method='higher')
        #q_l = np.quantile(S_l_cal, np.ceil((N+1)*(1-alpha_l))/N, method='higher')
        #q_u = np.quantile(S_u_cal, np.ceil((N+1)*(1-alpha_u))/N, method='higher')

        # --- build final intervals on test set ---
        intervals, preds = [], []
        for i in test_idx:
            lo, hi = quantile_band_asymm(rca_all[i], p_l, p_h)
            scale = hi - lo + 1e-3
            lb = max(0, lo - q_l*scale)
            ub = min(1, hi + q_u*scale)
            intervals.append((lb, ub))
            preds.append(np.mean(rca_all[i]))

        result = {
            'class': class_idx,
            'intervals': intervals,
            'y_test':   y_all[test_idx],
            'y_hat':    preds,
        }
        outputs.append(result)

    return outputs

