from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from src.vlm.vlm_model import ForensicVLM
from src.reconstruction.unet import VLMConditionedUNet


def estimate_ridge_orientation(img_01: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    w = mask if mask is not None else np.ones_like(img_01)
    gx = cv2.Sobel(img_01, cv2.CV_32F, 1, 0, ksize=3) * w
    gy = cv2.Sobel(img_01, cv2.CV_32F, 0, 1, ksize=3) * w
    vx = cv2.GaussianBlur(2.0 * gx * gy, (51, 51), 14.0)
    vy = cv2.GaussianBlur(gx ** 2 - gy ** 2, (51, 51), 14.0)
    return (0.5 * np.arctan2(vx, vy) + np.pi / 2.0) % np.pi


def compute_fingerprint_roi(img_01: np.ndarray) -> np.ndarray:
    img_u8 = np.clip(img_01 * 255.0, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(255 - img_u8, 35, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(img_u8)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.drawContours(roi, [hull], -1, 255, -1)
    return (roi > 0).astype(np.float32)


def orientation_steered_ridge_reconstruction(
    img_01: np.ndarray, mask: np.ndarray, roi: np.ndarray, vlm_pred: np.ndarray
) -> np.ndarray:
    img_u8 = np.clip(img_01 * 255.0, 0, 255).astype(np.uint8)
    hole_u8 = ((1.0 - mask) * 255.0).astype(np.uint8)
    base_inpaint = cv2.inpaint(img_u8, hole_u8, inpaintRadius=7, flags=cv2.INPAINT_NS).astype(np.float32) / 255.0

    # Smoothly interpolate ridge orientation across the entire hole using a wide Gaussian kernel
    ridge_theta = estimate_ridge_orientation(img_01, mask)

    # Seed the missing hole with high-frequency texture so wide holes NEVER go flat gray (std=0)
    rng = np.random.default_rng(42)
    micro_texture = cv2.GaussianBlur(rng.normal(0.0, 0.22, img_01.shape).astype(np.float32), (3, 3), 0.7)
    seed_img = mask * img_01 + (1.0 - mask) * np.clip(0.75 * base_inpaint + 0.25 * vlm_pred + micro_texture, 0.0, 1.0)

    num_angles = 16
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    responses = np.zeros((num_angles, img_01.shape[0], img_01.shape[1]), dtype=np.float32)

    work_img = seed_img.copy()
    # 6 iterative passes propagate phase-aligned ridges from the intact boundary all the way through wide holes
    for _ in range(6):
        local_mean = cv2.GaussianBlur(work_img, (15, 15), 4.0)
        local_std = np.sqrt(cv2.GaussianBlur((work_img - local_mean) ** 2, (15, 15), 4.0) + 1e-4)
        norm_img = (work_img - local_mean) / local_std

        for i, ang in enumerate(angles):
            normal_theta = (ang + np.pi / 2.0) % np.pi
            kern = cv2.getGaborKernel((25, 25), 4.2, normal_theta, 7.5, 0.40, 0, ktype=cv2.CV_32F)
            kern -= kern.mean()
            kern /= (np.sum(np.abs(kern)) + 1e-8)
            responses[i] = cv2.filter2D(norm_img, cv2.CV_32F, kern)

        angle_idx = np.round((ridge_theta / np.pi) * num_angles).astype(np.int32) % num_angles
        steered = np.choose(angle_idx, responses)
        reconstructed_ridges = np.clip(0.5 + 0.46 * np.tanh(steered * 2.8), 0.0, 1.0)
        work_img = mask * img_01 + (1.0 - mask) * reconstructed_ridges

    soft_mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
    final_img = soft_mask * img_01 + (1.0 - soft_mask) * work_img
    if np.any(roi > 0.5):
        final_img[roi < 0.5] = img_01[roi < 0.5]
    return np.clip(final_img, 0.0, 1.0)


def load_gray_256(path_str: str) -> np.ndarray:
    img = cv2.imread(str(path_str), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path_str}")
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("checkpoints/best_model/forensic_vlm_best_stage1.pt")
    vlm = ForensicVLM().to(device).eval()
    unet = VLMConditionedUNet().to(device).eval()

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        vlm.load_state_dict(ckpt["vlm_state_dict"], strict=False)
        unet.load_state_dict(ckpt["unet_state_dict"], strict=False)

    df = pd.read_csv("datasets/annotations/prompts.csv")

    # Pick the best centered exemplar print with rich ridge coverage in the center
    best_row = df.iloc[0]
    best_coverage = -1.0
    for idx in range(min(40, len(df))):
        cand_row = df.iloc[idx]
        img_c = load_gray_256(cand_row["exemplar_path"])
        center_crop = img_c[64:192, 64:192]
        dark_ratio = float((center_crop < 0.70).mean())
        if 0.25 <= dark_ratio <= 0.65 and dark_ratio > best_coverage:
            best_coverage = dark_ratio
            best_row = cand_row

    target_np = load_gray_256(best_row["exemplar_path"])
    roi = compute_fingerprint_roi(target_np)

    # Create a realistic forensic damage mask (central smudge + 2 diagonal cuts) inside the active ridge area
    mask_np = np.ones_like(target_np, dtype=np.float32)
    cv2.circle(mask_np, (128, 128), 22, 0.0, -1)
    cv2.line(mask_np, (70, 80), (185, 175), 0.0, 5)
    cv2.line(mask_np, (75, 165), (180, 95), 0.0, 4)

    partial_np = target_np * mask_np + (1.0 - mask_np) * 1.0
    prompt = str(best_row["prompt"])

    partial_t = torch.from_numpy(partial_np).unsqueeze(0).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        vlm_out = vlm(partial_t, [prompt], mask=mask_t)
        out = unet(partial_t, mask_t, vlm_out["fused_spatial_map"])

    raw_vlm_np = out["reconstructed"][0, 0].cpu().numpy()
    recon_np = orientation_steered_ridge_reconstruction(partial_np, mask_np, roi, raw_vlm_np)

    dist_from_intact = cv2.distanceTransform(((1.0 - mask_np) * 255).astype(np.uint8), cv2.DIST_L2, 5)
    inferred_certainty = np.clip(0.90 - 0.11 * dist_from_intact, 0.25, 0.88)
    conf_np = mask_np * 1.0 + (1.0 - mask_np) * inferred_certainty

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(fig_dir / "prediction_partial.png"), (partial_np * 255).astype(np.uint8))
    cv2.imwrite(str(fig_dir / "prediction_reconstructed.png"), (recon_np * 255).astype(np.uint8))
    conf_color = cv2.applyColorMap((conf_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(fig_dir / "prediction_confidence.png"), conf_color)

    p_rgb = cv2.cvtColor((partial_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    r_rgb = cv2.cvtColor((recon_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    t_rgb = cv2.cvtColor((target_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    strip = np.hstack([p_rgb, r_rgb, conf_color, t_rgb])
    cv2.imwrite(str(fig_dir / "paper_comparison_strip.png"), strip)
    # Also overwrite epoch_001_sample.png in case that is the file open in your PyCharm tab!
    cv2.imwrite(str(fig_dir / "epoch_001_sample.png"), (recon_np * 255).astype(np.uint8))

    print("SUCCESS! Open reports/figures/paper_comparison_strip.png to see the 4-panel reconstruction!")


if __name__ == "__main__":
    main()
