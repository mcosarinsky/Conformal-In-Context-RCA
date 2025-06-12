import numpy as np

def quantile_band_asymm(arr, p_l, p_h):
    """Asymmetric raw band: [p_l, p_h] quantiles of arr."""
    lo, hi = np.percentile(arr, [p_l * 100, p_h * 100])
    return lo, hi

import numpy as np

def quantile_calibrate(dataset_dict, p_l=0.40, p_h=0.95, alpha=0.10, n=50):
    real_cal = np.array(dataset_dict['cal']['Real score'])
    real_test = np.array(dataset_dict['test']['Real score'])
    rca_cal = np.array(dataset_dict['cal']['RCA score'])[:, :n]
    rca_test = np.array(dataset_dict['test']['RCA score'])[:, :n]

    is_mc = np.ndim(real_cal) == 2
    k = real_cal.shape[1] if is_mc else 1

    outputs = []
    for class_idx in range(k):
        y_cal = real_cal[:, class_idx] if is_mc else real_cal
        y_test = real_test[:, class_idx] if is_mc else real_test
        rca_cal_class = rca_cal[:, :, class_idx] if is_mc else rca_cal
        rca_test_class = rca_test[:, :, class_idx] if is_mc else rca_test

        S_cal = []
        for y, rca in zip(y_cal, rca_cal_class):
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            S_cal.append(max(lo - y, y - hi))

        N = len(S_cal)
        qhat = np.quantile(S_cal, np.ceil((N + 1) * (1 - alpha)) / N, method='higher')

        intervals = []
        preds = []
        for rca in rca_test_class:
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            lb = max(0, lo - qhat)
            ub = min(1, hi + qhat)
            intervals.append((lb, ub))
            preds.append(np.mean(rca))

        S_test = []
        for y, rca in zip(y_test, rca_test_class):
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            S_test.append(max(lo - y, y - hi))

        outputs.append({
            'class': class_idx,
            'intervals': intervals,
            'y_test': y_test,
            'y_hat': preds,
            'scores': S_cal,
            'test_scores': S_test,
            'q': qhat,
        })

    return outputs


def quantile_calibrate_asymm(dataset_dict, p_l=0.40, p_h=0.95, alpha_l=0.09, alpha_u=0.01, n=50):
    real_cal = np.array(dataset_dict['cal']['Real score'])
    real_test = np.array(dataset_dict['test']['Real score'])
    rca_cal = np.array(dataset_dict['cal']['RCA score'])[:, :n]
    rca_test = np.array(dataset_dict['test']['RCA score'])[:, :n]

    is_mc = np.ndim(real_cal) == 2
    k = real_cal.shape[1] if is_mc else 1

    outputs = []
    for class_idx in range(k):
        y_cal = real_cal[:, class_idx] if is_mc else real_cal
        y_test = real_test[:, class_idx] if is_mc else real_test
        rca_cal_class = rca_cal[:, :, class_idx] if is_mc else rca_cal
        rca_test_class = rca_test[:, :, class_idx] if is_mc else rca_test

        S_l, S_u = [], []
        for y, rca in zip(y_cal, rca_cal_class):
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            S_l.append(max(lo - y, 0))
            S_u.append(max(y - hi, 0))

        N = len(y_cal)
        q_l = np.quantile(S_l, min(np.ceil((N + 1) * (1 - alpha_l)) / N, 1.0), method='higher')
        q_u = np.quantile(S_u, min(np.ceil((N + 1) * (1 - alpha_u)) / N, 1.0), method='higher')

        intervals = []
        preds = []
        for rca in rca_test_class:
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            lb = max(0, lo - q_l)
            ub = min(1, hi + q_u)
            intervals.append((lb, ub))
            preds.append(np.mean(rca))

        outputs.append({
            'class': class_idx,
            'intervals': intervals,
            'y_test': y_test,
            'y_hat': preds,
        })

    return outputs


def quantile_calibrate_adaptative(dataset_dict, p_l=0.40, p_h=0.95, alpha=0.10, n=50):
    real_cal = np.array(dataset_dict['cal']['Real score'])
    real_test = np.array(dataset_dict['test']['Real score'])
    rca_cal = np.array(dataset_dict['cal']['RCA score'])[:, :n]
    rca_test = np.array(dataset_dict['test']['RCA score'])[:, :n]

    is_mc = np.ndim(real_cal) == 2
    k = real_cal.shape[1] if is_mc else 1
    eps = 1e-3

    outputs = []
    for class_idx in range(k):
        y_cal = real_cal[:, class_idx] if is_mc else real_cal
        y_test = real_test[:, class_idx] if is_mc else real_test
        rca_cal_class = rca_cal[:, :, class_idx] if is_mc else rca_cal
        rca_test_class = rca_test[:, :, class_idx] if is_mc else rca_test

        S_cal = []
        for y, rca in zip(y_cal, rca_cal_class):
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            scale = hi - lo + eps
            S_cal.append(max(lo - y, y - hi) / scale)

        N = len(y_cal)
        qhat = np.quantile(S_cal, np.ceil((N + 1) * (1 - alpha)) / N, method='higher')

        intervals = []
        preds = []
        for rca in rca_test_class:
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            scale = hi - lo + eps
            lb = max(0, lo - qhat * scale)
            ub = min(1, hi + qhat * scale)
            intervals.append((lb, ub))
            preds.append(np.mean(rca))

        outputs.append({
            'class': class_idx,
            'intervals': intervals,
            'y_test': y_test,
            'y_hat': preds,
        })

    return outputs


def quantile_calibrate_adaptative_asymm(dataset_dict, p_l=0.40, p_h=0.95, alpha_l=0.09, alpha_u=0.01, n=50):
    real_cal = np.array(dataset_dict['cal']['Real score'])
    real_test = np.array(dataset_dict['test']['Real score'])
    rca_cal = np.array(dataset_dict['cal']['RCA score'])[:, :n]
    rca_test = np.array(dataset_dict['test']['RCA score'])[:, :n]

    is_mc = np.ndim(real_cal) == 2
    k = real_cal.shape[1] if is_mc else 1
    eps = 1e-3

    outputs = []
    for class_idx in range(k):
        y_cal = real_cal[:, class_idx] if is_mc else real_cal
        y_test = real_test[:, class_idx] if is_mc else real_test
        rca_cal_class = rca_cal[:, :, class_idx] if is_mc else rca_cal
        rca_test_class = rca_test[:, :, class_idx] if is_mc else rca_test

        S_l_cal, S_u_cal = [], []
        for y, rca in zip(y_cal, rca_cal_class):
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            scale = hi - lo + eps
            S_l_cal.append(max(lo - y, 0) / scale)
            S_u_cal.append(max(y - hi, 0) / scale)

        N = len(y_cal)
        q_l = np.quantile(S_l_cal, min(np.ceil((N + 1) * (1 - alpha_l)) / N, 1.0), method='higher')
        q_u = np.quantile(S_u_cal, min(np.ceil((N + 1) * (1 - alpha_u)) / N, 1.0), method='higher')

        intervals = []
        preds = []
        for rca in rca_test_class:
            lo, hi = quantile_band_asymm(rca, p_l, p_h)
            scale = hi - lo + eps
            lb = max(0, lo - q_l * scale)
            ub = min(1, hi + q_u * scale)
            intervals.append((lb, ub))
            preds.append(np.mean(rca))

        outputs.append({
            'class': class_idx,
            'intervals': intervals,
            'y_test': y_test,
            'y_hat': preds,
        })

    return outputs

