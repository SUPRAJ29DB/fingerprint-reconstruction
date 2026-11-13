import yaml
from pathlib import Path
import torch
from torch.optim import AdamW
from tqdm import tqdm
from src.vlm.vlm_model import ForensicVLM
from src.reconstruction.unet import VLMConditionedUNet
from src.reconstruction.losses import ForensicReconstructionLoss
from src.training.dataset import create_dataloaders
from src.training.validate import evaluate_model

class ForensicVLMTrainer:
    def __init__(self, config_path: str = "configs/training.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        self.vlm = ForensicVLM(self.cfg["vlm_model_name"], self.cfg["embed_dim"], self.cfg["num_heads"], self.cfg["freeze_clip"]).to(self.device)
        self.unet = VLMConditionedUNet(2, self.cfg["embed_dim"], self.cfg["unet_base_ch"]).to(self.device)
        self.criterion = ForensicReconstructionLoss(self.cfg["lambda_hole"], self.cfg["lambda_orient"], self.cfg["lambda_uncert"]).to(self.device)
        params = [p for p in list(self.vlm.parameters()) + list(self.unet.parameters()) if p.requires_grad]
        self.optimizer = AdamW(params, lr=float(self.cfg["lr"]), weight_decay=float(self.cfg["weight_decay"]))
        self.train_loader, self.val_loader = create_dataloaders(
            self.cfg["train_csv"], self.cfg["stage"], self.cfg["image_size"], self.cfg["batch_size"], self.cfg["val_split"], self.cfg["num_workers"]
        )

    def train(self):
        best_ssim = -1.0
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.vlm.train()
            self.unet.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}/{self.cfg['epochs']}]")
            for batch in pbar:
                partial_img = batch["partial_img"].to(self.device)
                target_img = batch["target_img"].to(self.device)
                mask = batch["mask"].to(self.device)
                self.optimizer.zero_grad()
                vlm_out = self.vlm(partial_img, batch["prompt"], mask=mask)
                recon_out = self.unet(partial_img, mask, vlm_out["fused_spatial_map"])
                loss = self.criterion(recon_out, target_img, mask)["total_loss"]
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_metrics = evaluate_model(self.vlm, self.unet, self.val_loader, self.criterion, self.device, epoch, self.cfg["report_dir"])
            print(f"Epoch {epoch:02d} | Val PSNR: {val_metrics['psnr']:.2f} dB | Val SSIM: {val_metrics['ssim']:.4f} | Conf: {val_metrics['mean_inferred_confidence']:.3f}")
            if val_metrics["ssim"] > best_ssim:
                best_ssim = val_metrics["ssim"]
                ckpt_path = Path(self.cfg["checkpoint_dir"]) / f"forensic_vlm_best_{self.cfg['stage']}.pt"
                torch.save({"vlm_state_dict": self.vlm.state_dict(), "unet_state_dict": self.unet.state_dict()}, ckpt_path)
