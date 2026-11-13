import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FingerprintVisionEncoder(nn.Module):
    def __init__(self, clip_vision_model: nn.Module, out_dim: int = 512):
        super().__init__()
        self.vision_model = clip_vision_model
        self.hidden_size = clip_vision_model.config.hidden_size
        self.proj = nn.Linear(self.hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> dict:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        outputs = self.vision_model(pixel_values=x, output_hidden_states=True, return_dict=True)
        seq_tokens = self.norm(self.proj(outputs.last_hidden_state))
        cls_token = seq_tokens[:, 0, :]
        patch_tokens = seq_tokens[:, 1:, :]
        B, N, D = patch_tokens.shape
        grid_size = int(math.isqrt(N))
        spatial_map = patch_tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        return {"patch_tokens": patch_tokens, "spatial_map": spatial_map, "cls_token": cls_token}
