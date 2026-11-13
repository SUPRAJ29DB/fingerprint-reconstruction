import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def compute_forensic_psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> dict:
    global_psnr = float(peak_signal_noise_ratio(target, pred, data_range=1.0))
    hole_psnr = global_psnr
    if mask is not None and np.any(mask < 0.5):
        mse_hole = np.mean((target[mask < 0.5] - pred[mask < 0.5]) ** 2)
        hole_psnr = float(10.0 * np.log10(1.0 / (mse_hole + 1e-10)))
    return {"psnr_global": global_psnr, "psnr_hole": hole_psnr}
