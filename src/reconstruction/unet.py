import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class VLMConditionedUNet(nn.Module):
    def __init__(self, in_channels: int = 2, vlm_dim: int = 512, base_ch: int = 64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 8 + vlm_dim, base_ch * 8)
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 4)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.recon_head = nn.Sequential(nn.Conv2d(base_ch, 1, kernel_size=1), nn.Sigmoid())
        self.uncertainty_head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, partial_img: torch.Tensor, mask: torch.Tensor, vlm_spatial_map: torch.Tensor) -> dict:
        x = torch.cat([partial_img, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        p4 = self.pool(e4)
        vlm_feat = F.interpolate(vlm_spatial_map, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        b = self.bottleneck(torch.cat([p4, vlm_feat], dim=1))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        raw_recon = self.recon_head(d1)
        blended_recon = mask * partial_img + (1.0 - mask) * raw_recon
        log_var = torch.clamp(self.uncertainty_head(d1), min=-6.0, max=4.0)
        inferred_confidence = torch.exp(-torch.exp(log_var))
        confidence_map = torch.clamp(mask + (1.0 - mask) * inferred_confidence, 0.0, 1.0)
        return {"reconstructed": blended_recon, "raw_reconstructed": raw_recon, "log_var": log_var, "confidence_map": confidence_map}
