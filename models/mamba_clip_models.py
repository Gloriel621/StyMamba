# Based on CLIP code bases
# Modified from github.com/openai/CLIP
# --------------------------------------------------------'

from collections import OrderedDict
import numpy as np
import timm
import torch
import os
import sys
from torch import nn
from argparse import Namespace

from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from VMamba2.classification.models.mamba import Mamba2

def get_att_mask(attention, ratio=0.5):
    bs = attention.shape[0]
    masks = torch.ones((bs, 49), dtype=torch.bool, device=attention.device)
    attention = attention.reshape((-1, 14, 14))
    attention = torch.nn.functional.interpolate(
        attention.unsqueeze(1), (7, 7), mode="bilinear"
    ).squeeze()
    attention = attention.reshape(bs, -1)
    N = int(attention.shape[1] * ratio)

    reservation = torch.argsort(attention, descending=True)
    reservation = reservation[:, : N + 1]
    masks = masks.scatter_(1, reservation, False)

    full_mask = torch.zeros((bs, 14, 14), dtype=torch.bool, device=attention.device)
    full_mask[:, 0::2, 0::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 0::2, 1::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 1::2, 0::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 1::2, 1::2] = masks.reshape(bs, 7, 7)
    full_mask = full_mask.reshape(bs, -1)

    return full_mask


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        vision_width: int,
        vision_model: nn.Module,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.mamba = Mamba2(
            d_model=512,
            d_state=64,
            d_conv=4,
            expand=2,
        ).to("cuda")

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.ln_final_e = LayerNorm(transformer_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        nn.init.normal_(self.image_projection, std=self.vision_width**-0.5)
        nn.init.normal_(self.text_projection, std=0.01)

    def build_attention_mask(self):

        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection

        return x

    def encode_text_mamba(self, text, ema=False):

        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.mamba(x)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.text_projection

        return x

    def forward(self, image, text, momentum=0):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text_mamba(text)

        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),
        }

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/home/gloriel621/stymamba/VMamba2/classification')

def CLIP_VMamba_S(**kwargs):

    from VMamba2.classification.models import build_model
    from config import get_config

    args = Namespace(**kwargs)
    cfg = get_config(args)
    vision_model = build_model(cfg)

    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )
    return model
