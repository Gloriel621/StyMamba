import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import copy

from timm.models.layers import DropPath
from VMamba2.classification.models.vmamba import VSSBlock, VSSBlockOneWay, Mlp, mamba_init
from VMamba2.classification.models.csm_triton import cross_scan_fn
from VMamba2.classification.models.csms6s import selective_scan_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CrossModalIntegrationLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        drop_path: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ssm_d_state: int = 64,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: any = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        mlp_act_layer: nn.Module = nn.GELU,
        mlp_drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.d_inner = int(ssm_ratio * hidden_dim)
        self.ssm_d_state = ssm_d_state
        self.ssm_dt_rank = int(math.ceil(hidden_dim / 16)) if ssm_dt_rank == "auto" else ssm_dt_rank
        self.k_group = 4

        self.norm1 = norm_layer(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, self.d_inner * 2, bias=False)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=True, kernel_size=ssm_conv,
            padding=(ssm_conv - 1) // 2,
        )
        
        self.style_proj = nn.Linear(hidden_dim, self.k_group * (self.ssm_dt_rank + self.ssm_d_state), bias=False)
        self.content_proj = nn.Linear(self.d_inner, self.k_group * self.ssm_d_state, bias=False)
        
        dt_projs = [mamba_init.dt_init(self.ssm_dt_rank, self.d_inner) for _ in range(self.k_group)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
        
        self.A_logs = nn.Parameter(torch.zeros(self.k_group * self.d_inner, self.ssm_d_state))
        self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, hidden_dim, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(hidden_dim)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate,
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor):
        B, H, W, C = content.shape
        L = H * W
        
        residual_1 = content
        
        content_norm = self.norm1(content)
        style_norm = self.norm1(style)

        x_z = self.in_proj(content_norm)
        x, z = x_z.chunk(2, dim=-1)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x_conv = self.conv2d(x)
        
        style_params = self.style_proj(style_norm)
        dts_style_rank, Bs_style = torch.split(style_params, [self.k_group * self.ssm_dt_rank, self.k_group * self.ssm_d_state], dim=-1)
        
        Cs_content = self.content_proj(self.act(x_conv).permute(0, 2, 3, 1).contiguous())
        
        xs = cross_scan_fn(self.act(x_conv), in_channel_first=True)
        dts_rank = dts_style_rank.view(B, L, self.k_group, self.ssm_dt_rank).permute(0, 2, 3, 1)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts_rank, self.dt_projs_weight)
        
        xs = xs.flatten(1, 2)
        dts = dts.flatten(1, 2)
        
        Bs = Bs_style.view(B, L, self.k_group, self.ssm_d_state).permute(0, 2, 3, 1).contiguous()
        Cs = Cs_content.view(B, L, self.k_group, self.ssm_d_state).permute(0, 2, 3, 1).contiguous()
        
        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        delta_bias = self.dt_projs_bias.flatten()
        
        y_scan = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds,
            delta_bias=delta_bias,
            delta_softplus=True,
        )
        
        y_scan = y_scan.view(B, self.k_group, self.d_inner, H, W)

        y = y_scan.sum(dim=1)
        y = y.permute(0, 2, 3, 1).contiguous()
        
        y = y * z
        y = self.out_proj(y)
        
        x = residual_1 + self.drop_path(y)
        
        residual_2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        final_output = residual_2 + self.drop_path(x)
        
        return final_output

class MambaNet(nn.Module):
    def __init__(self, d_model=512, num_decoder_layers=1, dropout=0.1, drop_path=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalIntegrationLayer(hidden_dim=d_model, dropout=dropout, drop_path=drop_path, **kwargs)
            for _ in range(num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self._reset_parameters()

        # Positional embedding components
        self.new_ps = nn.Conv2d(d_model, d_model, (1, 1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style, mask, content, pos_embed_c, pos_embed_s):

        content_pool = self.averagepooling(content)
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=content.shape[-2:])
        
        content = content.permute(0, 2, 3, 1).contiguous()
        style = style.permute(0, 2, 3, 1).contiguous()
        
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.permute(0, 2, 3, 1).contiguous()
            content = content + pos_embed_c

        output = content
        for layer in self.layers:
            output = layer(output, style)
        
        output = self.norm(output)
        hs = output.permute(0, 3, 1, 2).contiguous()

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

class Mamba_TransFormer(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super(Mamba_TransFormer, self).__init__()
        self.hidden_dim = hidden_dim
        self.vss_block_content = VSSBlock(hidden_dim=hidden_dim, drop_path=dropout, ssm_d_state=64, ssm_act_layer=nn.GELU, ssm_init="v2", forward_type="m0_noz")
        # for style, use one way scan.
        self.vss_block_style = self.vss_block = VSSBlock(hidden_dim=hidden_dim, drop_path=dropout, ssm_d_state=64, ssm_act_layer=nn.GELU, ssm_init="v2", forward_type="m0_noz")
        self.combine_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        
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

    def forward(self, content, style):

        q = content
        k = style
        v = style

        tgt2 = self.vss_block_content(q)
        tgt = content + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2_style = self.vss_block_style(k)
        tgt2 = v + self.dropout2(tgt2_style)
        tgt = tgt + self.norm2(tgt2)
        
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

    def forward(self, tgt, memory):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory)
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
