import torch
import torch.nn as nn
from typing import List, Union

class ForensicTextEncoder(nn.Module):
    def __init__(self, clip_text_model: nn.Module, tokenizer, out_dim: int = 512):
        super().__init__()
        self.text_model = clip_text_model
        self.tokenizer = tokenizer
        self.hidden_size = clip_text_model.config.hidden_size
        self.proj = nn.Linear(self.hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, prompts: Union[List[str], torch.Tensor], device: torch.device) -> dict:
        if isinstance(prompts, list):
            tokens = self.tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
        else:
            input_ids = prompts.to(device)
            attention_mask = None
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        token_embeds = self.norm(self.proj(outputs.last_hidden_state))
        pooled_embed = self.norm(self.proj(outputs.pooler_output))
        return {"token_embeds": token_embeds, "pooled_embed": pooled_embed, "attention_mask": attention_mask}
