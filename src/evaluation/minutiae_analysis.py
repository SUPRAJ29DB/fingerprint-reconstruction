import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def extract_minutiae(img: np.ndarray, border_margin: int = 16) -> np.ndarray:
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_u8)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    skeleton = skeletonize(binary > 0).astype(np.uint8)
    H, W = skeleton.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    neighbors = [np.roll(np.roll(skeleton, dy, axis=0), dx, axis=1) for dy, dx in offsets]
    cn = sum(np.abs(neighbors[i].astype(np.float32) - neighbors[(i + 1) % 8].astype(np.float32)) for i in range(8))
    cn = (0.5 * cn).astype(np.int32) * skeleton
    valid = np.zeros_like(skeleton, dtype=bool)
    valid[border_margin : H - border_margin, border_margin : W - border_margin] = True
    ey, ex = np.where((cn == 1) & valid)
    by, bx = np.where((cn == 3) & valid)
    endings = np.column_stack([ey, ex, np.ones_like(ey)])
    bifurc = np.column_stack([by, bx, np.full_like(by, 3)])
    return np.vstack([endings, bifurc]) if len(endings) + len(bifurc) > 0 else np.empty((0, 3))

def evaluate_minutiae_preservation(pred: np.ndarray, target: np.ndarray, distance_threshold: float = 12.0) -> dict:
    m_pred, m_targ = extract_minutiae(pred), extract_minutiae(target)
    if len(m_targ) == 0 or len(m_pred) == 0:
        return {"minutiae_preservation_rate": 0.0, "minutiae_precision": 0.0, "minutiae_localization_error": float(distance_threshold), "match_similarity_score": 0.0}
    dist_matrix = cdist(m_pred[:, :2], m_targ[:, :2], metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matched = [dist_matrix[r, c] for r, c in zip(row_ind, col_ind) if dist_matrix[r, c] <= distance_threshold]
    recall = len(matched) / max(1, len(m_targ))
    prec = len(matched) / max(1, len(m_pred))
    return {
        "minutiae_preservation_rate": float(recall),
        "minutiae_precision": float(prec),
        "minutiae_localization_error": float(np.mean(matched)) if matched else float(distance_threshold),
        "match_similarity_score": float((2 * recall * prec) / (recall + prec + 1e-8)),
    }
