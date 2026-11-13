import cv2
import numpy as np

def compute_orientation_field(img: np.ndarray, block_size: int = 16) -> np.ndarray:
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.boxFilter(gx * gx, -1, (block_size, block_size))
    gyy = cv2.boxFilter(gy * gy, -1, (block_size, block_size))
    gxy = cv2.boxFilter(gx * gy, -1, (block_size, block_size))
    return 0.5 * np.arctan2(2.0 * gxy, gxx - gyy + 1e-8)

def compute_ridge_continuity_score(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None, block_size: int = 16) -> dict:
    alignment_map = 0.5 * (1.0 + np.cos(2.0 * (compute_orientation_field(pred, block_size) - compute_orientation_field(target, block_size))))
    global_rocs = float(np.mean(alignment_map))
    hole_rocs = float(np.mean(alignment_map[mask < 0.5])) if (mask is not None and np.any(mask < 0.5)) else global_rocs
    return {"ridge_continuity_global": global_rocs, "ridge_continuity_hole": hole_rocs}
