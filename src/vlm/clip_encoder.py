import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor

class CLIPBackbone(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", freeze_backbone: bool = True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad = False

    @property
    def vision_model(self):
        return self.clip.vision_model

    @property
    def text_model(self):
        return self.clip.text_model
