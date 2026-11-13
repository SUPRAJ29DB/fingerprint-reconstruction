import math
import torch
import torch.nn as nn

class CrossModalForensicFusion(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.film_generator = nn.Linear(embed_dim, embed_dim * 2)

    def forward(self, patch_tokens: torch.Tensor, text_tokens: torch.Tensor, pooled_text: torch.Tensor, text_mask: torch.Tensor = None) -> dict:
        key_padding_mask = (text_mask == 0) if text_mask is not None else None
        attn_out, attn_weights = self.cross_attn(
            query=patch_tokens, key=text_tokens, value=text_tokens,
            key_padding_mask=key_padding_mask, need_weights=True
        )
        x = self.norm1(patch_tokens + attn_out)
        gamma, beta = self.film_generator(pooled_text).chunk(2, dim=-1)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        fused_tokens = self.norm2(x + self.ffn(x))
        B, N, D = fused_tokens.shape
        grid_size = int(math.isqrt(N))
        fused_spatial_map = fused_tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        return {"fused_tokens": fused_tokens, "fused_spatial_map": fused_spatial_map, "attn_weights": attn_weights}
