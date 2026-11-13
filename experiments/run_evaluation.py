from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from src.vlm.vlm_model import ForensicVLM
from src.reconstruction.unet import VLMConditionedUNet
from src.training.dataset import ForensicFingerprintDataset
from src.evaluation.psnr import compute_forensic_psnr
from src.evaluation.ssim import compute_forensic_ssim
from src.evaluation.ridge_analysis import compute_ridge_continuity_score
from src.evaluation.minutiae_analysis import evaluate_minutiae_preservation
from src.evaluation.uncertainty import evaluate_forensic_uncertainty

@torch.no_grad()
def run_full_benchmark(
    checkpoint_path: str = "checkpoints/best_model/forensic_vlm_best_stage1.pt",
    test_csv: str = "datasets/annotations/prompts.csv",
    stage: str = "stage1",
    output_report_csv: str = "reports/metrics/benchmark_results.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    vlm = ForensicVLM().to(device).eval()
    unet = VLMConditionedUNet().to(device).eval()
    vlm.load_state_dict(ckpt["vlm_state_dict"])
    unet.load_state_dict(ckpt["unet_state_dict"])

    dataset = ForensicFingerprintDataset(csv_path=test_csv, stage=stage)
    records = []
    for idx in tqdm(range(len(dataset)), desc=f"Evaluating ({stage})"):
        sample = dataset[idx]
        p_img = sample["partial_img"].unsqueeze(0).to(device)
        t_img = sample["target_img"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        vlm_out = vlm(p_img, [sample["prompt"]], mask=mask)
        recon_out = unet(p_img, mask, vlm_out["fused_spatial_map"])

        pred_np = recon_out["reconstructed"][0, 0].cpu().numpy()
        targ_np = t_img[0, 0].cpu().numpy()
        mask_np = mask[0, 0].cpu().numpy()
        conf_np = recon_out["confidence_map"][0, 0].cpu().numpy()

        m = {}
        m.update(compute_forensic_psnr(pred_np, targ_np, mask_np))
        m.update(compute_forensic_ssim(pred_np, targ_np, mask_np))
        m.update(compute_ridge_continuity_score(pred_np, targ_np, mask_np))
        m.update(evaluate_minutiae_preservation(pred_np, targ_np))
        m.update(evaluate_forensic_uncertainty(pred_np, targ_np, conf_np, mask_np))
        records.append(m)

    Path(output_report_csv).parent.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(records)
    df_res.to_csv(output_report_csv, index=False)
    print("\n=== ForensicVLM Quantitative Evaluation Summary ===")
    for col in df_res.columns:
        print(f"{col:30s}: {df_res[col].mean():.4f} +/- {df_res[col].std():.4f}")
