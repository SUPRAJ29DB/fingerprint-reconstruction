import random
import cv2
import numpy as np

REGIONS = {
    "upper-left": (0.0, 0.5, 0.0, 0.5),
    "upper-right": (0.0, 0.5, 0.5, 1.0),
    "lower-left": (0.5, 1.0, 0.0, 0.5),
    "lower-right": (0.5, 1.0, 0.5, 1.0),
    "central core": (0.25, 0.75, 0.25, 0.75),
}

def generate_forensic_mask_and_prompt(img_shape: tuple = (256, 256), finger_name: str = "fingerprint", occlusion_ratio: float = 0.4):
    H, W = img_shape
    mask = np.ones((H, W), dtype=np.float32)
    region_name, (r_y1, r_y2, r_x1, r_x2) = random.choice(list(REGIONS.items()))
    center = (random.randint(int(r_x1 * W) + 15, int(r_x2 * W) - 15), random.randint(int(r_y1 * H) + 15, int(r_y2 * H) - 15))
    axes = (int(W * 0.25 * (occlusion_ratio / 0.4)), int(H * 0.25 * (occlusion_ratio / 0.4)))
    cv2.ellipse(mask, center, axes, random.randint(0, 180), 0, 360, 0.0, -1)
    actual_missing_pct = int((1.0 - mask.mean()) * 100)
    prompt = f"Forensic {finger_name} with {actual_missing_pct}% missing ridges in {region_name}. Reconstruct continuous ridge flow and preserve minutiae."
    return mask, prompt
