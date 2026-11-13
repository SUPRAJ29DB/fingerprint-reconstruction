import torch
import torch.nn as nn
import torch.nn.functional as F

class RidgeOrientationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _compute_orientation_vector(self, img: torch.Tensor):
        gx = F.conv2d(img, self.sobel_x, padding=1)
        gy = F.conv2d(img, self.sobel_y, padding=1)
        vx = gx ** 2 - gy ** 2
        vy = 2 * gx * gy
        norm = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)
        return vx / norm, vy / norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vx_p, vy_p = self._compute_orientation_vector(pred)
        vx_t, vy_t = self._compute_orientation_vector(target)
        return F.l1_loss(vx_p, vx_t) + F.l1_loss(vy_p, vy_t)

class ForensicReconstructionLoss(nn.Module):
    def __init__(self, lambda_hole: float = 5.0, lambda_orient: float = 0.5, lambda_uncert: float = 0.2):
        super().__init__()
        self.lambda_hole = lambda_hole
        self.lambda_orient = lambda_orient
        self.lambda_uncert = lambda_uncert
        self.orient_loss = RidgeOrientationLoss()

    def forward(self, outputs: dict, target: torch.Tensor, mask: torch.Tensor) -> dict:
        pred = outputs["raw_reconstructed"]
        log_var = outputs["log_var"]
        l1_valid = torch.mean(torch.abs(mask * (pred - target)))
        l1_hole = torch.mean(torch.abs((1.0 - mask) * (pred - target)))
        precision = torch.exp(-log_var)
        uncert_loss = torch.mean(precision * torch.abs(pred - target) + 0.5 * log_var)
        orient_loss = self.orient_loss(outputs["reconstructed"], target)
        total_loss = l1_valid + self.lambda_hole * l1_hole + self.lambda_orient * orient_loss + self.lambda_uncert * uncert_loss
        return {"total_loss": total_loss, "l1_hole": l1_hole.detach(), "orient_loss": orient_loss.detach(), "uncert_loss": uncert_loss.detach()}
