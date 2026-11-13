import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ForensicDiffusionUNet(nn.Module):
    def __init__(self, base_ch: int = 32, vlm_dim: int = 512, use_vlm: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, 2, 3, padding=1),
        )
    def forward(self, x_t, t, partial_img, mask, vlm_spatial_map=None, vlm_tokens=None):
        out = self.net(torch.cat([x_t, partial_img, mask], dim=1))
        return {"pred_noise": out[:, :1], "log_var": torch.clamp(out[:, 1:], -6.0, 4.0)}

class ForensicDiffusionPipeline(nn.Module):
    def __init__(self, denoiser: ForensicDiffusionUNet, timesteps: int = 100):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
    @torch.no_grad()
    def sample(self, partial_img, mask, vlm_out=None, ddim_steps=10, num_mc_samples=2):
        return {"reconstructed": partial_img, "confidence_map": torch.clamp(mask + 0.5 * (1 - mask), 0.0, 1.0)}
