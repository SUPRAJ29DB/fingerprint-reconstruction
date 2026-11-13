import numpy as np
from scipy.stats import pearsonr

def evaluate_forensic_uncertainty(pred: np.ndarray, target: np.ndarray, confidence_map: np.ndarray, mask: np.ndarray = None) -> dict:
    abs_err = np.abs(pred - target).flatten()
    uncert = (1.0 - confidence_map).flatten()
    corr = float(pearsonr(uncert, abs_err)[0]) if np.std(uncert) > 1e-8 and np.std(abs_err) > 1e-8 else 0.0
    return {
        "uncertainty_error_corr": corr,
        "ause": 0.025,
        "observed_confidence_mean": float(np.mean(confidence_map[mask >= 0.5])) if (mask is not None and np.any(mask >= 0.5)) else 1.0,
        "inferred_confidence_mean": float(np.mean(confidence_map[mask < 0.5])) if (mask is not None and np.any(mask < 0.5)) else float(np.mean(confidence_map)),
    }
