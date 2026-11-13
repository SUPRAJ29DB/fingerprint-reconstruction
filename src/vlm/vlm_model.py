import torch
import torch.nn as nn
from typing import List, Union
from .clip_encoder import CLIPBackbone
from .vision_encoder import FingerprintVisionEncoder
from .text_encoder import ForensicTextEncoder
from .fusion import CrossModalForensicFusion

class ForensicVLM(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", embed_dim: int = 512, num_heads: int = 8, freeze_clip: bool = True):
        super().__init__()
        self.backbone = CLIPBackbone(model_name=model_name, freeze_backbone=freeze_clip)
        self.vision_encoder = FingerprintVisionEncoder(self.backbone.vision_model, out_dim=embed_dim)
        self.text_encoder = ForensicTextEncoder(self.backbone.text_model, self.backbone.tokenizer, out_dim=embed_dim)
        self.fusion = CrossModalForensicFusion(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, partial_image: torch.Tensor, prompts: Union[List[str], torch.Tensor], mask: torch.Tensor = None) -> dict:
        device = partial_image.device
        vis_out = self.vision_encoder(partial_image)
        txt_out = self.text_encoder(prompts, device=device)
        fused_out = self.fusion(vis_out["patch_tokens"], txt_out["token_embeds"], txt_out["pooled_embed"], txt_out["attention_mask"])
        return {
            "fused_tokens": fused_out["fused_tokens"],
            "fused_spatial_map": fused_out["fused_spatial_map"],
            "attn_weights": fused_out["attn_weights"],
            "pooled_text": txt_out["pooled_embed"],
        }
