from pathlib import Path
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

@torch.no_grad()
def evaluate_model(vlm_model, unet_model, val_loader, criterion, device, epoch: int = 0, report_dir: str = "reports/figures") -> dict:
    vlm_model.eval()
    unet_model.eval()
    total_loss, psnr_scores, ssim_scores, hole_conf_scores = 0.0, [], [], []
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    saved_visuals = False

    for batch in val_loader:
        partial_img = batch["partial_img"].to(device)
        target_img = batch["target_img"].to(device)
        mask = batch["mask"].to(device)
        vlm_out = vlm_model(partial_img, batch["prompt"], mask=mask)
        recon_out = unet_model(partial_img, mask, vlm_out["fused_spatial_map"])
        loss_dict = criterion(recon_out, target_img, mask)
        total_loss += loss_dict["total_loss"].item()

        pred_np = recon_out["reconstructed"].float().cpu().numpy()
        target_np = target_img.float().cpu().numpy()
        partial_np = partial_img.float().cpu().numpy()
        conf_np = recon_out["confidence_map"].float().cpu().numpy()
        mask_np = mask.float().cpu().numpy()

        for b in range(pred_np.shape[0]):
            psnr_scores.append(compute_psnr(target_np[b, 0], pred_np[b, 0], data_range=1.0))
            ssim_scores.append(compute_ssim(target_np[b, 0], pred_np[b, 0], data_range=1.0))
            if (mask_np[b, 0] < 0.5).any():
                hole_conf_scores.append(float(conf_np[b, 0][mask_np[b, 0] < 0.5].mean()))

        if not saved_visuals:
            to_u8 = lambda x: np.clip(x * 255.0, 0, 255).astype(np.uint8)
            grid = np.hstack([
                cv2.cvtColor(to_u8(partial_np[0, 0]), cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(to_u8(pred_np[0, 0]), cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(to_u8(target_np[0, 0]), cv2.COLOR_GRAY2BGR),
                cv2.applyColorMap(to_u8(conf_np[0, 0]), cv2.COLORMAP_JET),
            ])
            cv2.imwrite(f"{report_dir}/epoch_{epoch:03d}_sample.png", grid)
            saved_visuals = True

    return {
        "val_loss": total_loss / max(1, len(val_loader)),
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
        "mean_inferred_confidence": float(np.mean(hole_conf_scores)) if hole_conf_scores else 1.0,
    }
