import os
import sys
import copy
from typing import Optional
from transformers import AutoTokenizer
from VMamba2.classification.models.vmamba import VSSBlock, VSSBlockOneWay

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mamba_ssm_.models.mixer_seq_simple import MambaLMHeadModel

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MambaNet(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer_c = VSSEncoderLayer(hidden_dim=d_model, dropout=dropout)
        encoder_layer_s = VSSEncoderLayer(hidden_dim=d_model, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_c = MambaEncoder(encoder_layer_c, num_encoder_layers, encoder_norm)
        self.encoder_s = MambaEncoder(encoder_layer_s, num_encoder_layers, encoder_norm)

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        self.encoder_t = MambaLMHeadModel.from_pretrained(
            "state-spaces/mamba-130m",
            device=device,
            dtype=torch.float32
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.style_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        decoder_layer = VSSDecoderLayer(hidden_dim=d_model, dropout=dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MambaDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                    return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.new_ps = nn.Conv2d(512, 512, (1, 1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_image_style(self, style, mask, content, pos_embed_c, pos_embed_s):
        """
        Forward pass when the style input is an image.
        """
        # Content-aware positional embedding
        content_pool = self.averagepooling(content)
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=style.shape[-2:])

        # Flatten NxCxHxW to HWxNxC and ensure 4D tensors
        style = style.flatten(2).permute(2, 0, 1).unsqueeze(2)
        if pos_embed_s is not None:
            pos_embed_s = pos_embed_s.flatten(2).permute(2, 0, 1).unsqueeze(2)

        content = content.flatten(2).permute(2, 0, 1).unsqueeze(2)
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1).unsqueeze(2)

        # Encode style and content
        style = self.encoder_s(style, src_key_padding_mask=mask, pos=pos_embed_s)
        content = self.encoder_c(content, src_key_padding_mask=mask, pos=pos_embed_c)

        hs = self.decoder(content, style, pos_embed_s, pos_embed_c)[0]

        # Ensure hs is 3D before unpacking
        hs = hs.squeeze(2)

        # HWxNxC to NxCxHxW
        N, B, C = hs.shape
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0)
        hs = hs.view(B, C, -1, H)

        return hs

    def forward_text_style(self, text_input, mask, content, pos_embed_c):
        """
        Forward pass when the style input is text.
        """
        tokens = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens.input_ids.to(next(self.encoder_t.parameters()).device)  # Move to model's device
        # attention_mask = tokens.attention_mask.to(next(self.encoder_t.parameters()).device)

        # Content-aware positional embedding
        content_pool = self.averagepooling(content)
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=(content.size(-2), content.size(-1)))

        # Flatten NxCxHxW to HWxNxC and ensure 4D tensors
        content = content.flatten(2).permute(2, 0, 1).unsqueeze(2)
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1).unsqueeze(2)

        style = self.encoder_t.encode_text(input_ids=input_ids)

        style = self.style_projection(style)
        style = style.unsqueeze(1)
        style = style.repeat(512, 1, 1, 1)

        content = self.encoder_c(content, src_key_padding_mask=mask, pos=pos_embed_c)

        hs = self.decoder(content, style, None, pos_embed_c)[0]

        # Ensure hs is 3D before unpacking
        hs = hs.mean(dim=2) 

        # HWxNxC to NxCxHxW
        N, B, C = hs.shape
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0)
        hs = hs.view(B, C, -1, H)

        return hs

class VSSEncoderLayer(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super(VSSEncoderLayer, self).__init__()
        self.vss_block = VSSBlock(hidden_dim=hidden_dim, drop_path=dropout, ssm_d_state=64, ssm_act_layer=nn.GELU, ssm_init="v2", forward_type="m0_noz")

    def forward(self, src):
        return self.vss_block(src)
    
class VSSOnewayLayer(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super(VSSOnewayLayer, self).__init__()
        self.vss_block = VSSBlockOneWay(hidden_dim=hidden_dim, drop_path=dropout, ssm_d_state=64, ssm_act_layer=nn.GELU, ssm_init="v2", forward_type="m0_noz")
        
    def forward(self, src):
        return self.vss_block(src)

class VSSDecoderLayer(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super(VSSDecoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.vss_block_content = VSSBlock(hidden_dim=hidden_dim, drop_path=dropout, ssm_d_state=64, ssm_act_layer=nn.GELU, ssm_init="v2", forward_type="m0_noz")
        # for style, use one way scan.
        self.vss_block_style = self.vss_block = VSSBlock(hidden_dim=hidden_dim, drop_path=dropout, ssm_d_state=64, ssm_act_layer=nn.GELU, ssm_init="v2", forward_type="m0_noz")
        
        # FFN
        self.linear1 = nn.Linear(hidden_dim, 2048)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2048, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, content, style, pos, query_pos):

        # since pos = None, the style is not encoded with positional embedding
        q = self.with_pos_embed(content, query_pos)
        k = self.with_pos_embed(style, pos)
        v = style

        # Self-attention using VSSBlock for content
        tgt2 = self.vss_block_content(q)
        tgt = content + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention using VSSBlock for style (handling values)
        # Here, we treat the style sequence (k) as the input for cross-attention.
        # We pass the style sequence through the VSSBlock, and then combine it with the original content sequence.
        tgt2_style = self.vss_block_style(k)
        tgt2 = v + self.dropout2(tgt2_style)  # Integrate the value tensor here appropriately.
        tgt = tgt + self.norm2(tgt2)
        
        # Feedforward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class MambaEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output

class MambaDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, pos, query_pos):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_mambanet(args):
    return MambaNet(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
