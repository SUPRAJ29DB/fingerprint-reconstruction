import numpy as np
from skimage.metrics import structural_similarity

def compute_forensic_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> dict:
    global_ssim, ssim_map = structural_similarity(target, pred, data_range=1.0, full=True)
    hole_ssim = float(np.mean(ssim_map[mask < 0.5])) if (mask is not None and np.any(mask < 0.5)) else float(global_ssim)
    return {"ssim_global": float(global_ssim), "ssim_hole": hole_ssim}
