"""
Reusable KBBlock building block with enhanced KBA attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from .kb_utils import KBAFunction, LayerNorm2d, SimpleGate
from .dropblock import DropBlock2D

__all__ = ["KBBlock", "UNetBlock"]


def _init_weight(tensor):
    if tensor is None:
        return
    init.kaiming_uniform_(tensor, a=math.sqrt(5))


class KBBlock(nn.Module):
    """Knowledge-Based attention block with multi-branch attention and hybrid convolution."""

    def __init__(
        self,
        c: int,
        dw_expand: float = 2.0,
        ffn_expand: float = 2.0,
        nset: int = 32,
        kernel_size: int = 3,
        groups_per_channel: int = 4,
        lightweight: bool = False,
        dropblock_prob: float = 0.0,
        dropblock_size: int = 7,
        topk: int = 0,
        low_rank_ratio: float = 0.25,
        alpha_init: float = 0.5,
        global_ratio: float = 0.5,
        stage_id: int = 0,
        disable_local_attention: bool = False,
        disable_global_attention: bool = False,
        disable_dynamic_conv: bool = False,
        disable_static_conv: bool = False,
    ):
        super().__init__()
        self.k = kernel_size
        self.c = c
        self.nset = nset
        self.topk = topk if topk is not None else 0
        self.stage_id = stage_id
        self.disable_local_attention = disable_local_attention
        self.disable_global_attention = disable_global_attention
        self.disable_dynamic_conv = disable_dynamic_conv
        self.disable_static_conv = disable_static_conv
        dw_ch = int(c * dw_expand)
        ffn_ch = int(ffn_expand * c)

        self.g = max(1, c // max(groups_per_channel, 1))
        out_dim = c * c // self.g * self.k ** 2
        rank_limit = self.nset * 4
        rank = max(1, int(out_dim * low_rank_ratio))
        if rank_limit > 0:
            rank = min(rank, rank_limit)
        self.rank = rank

        self.w_coeff = nn.Parameter(torch.zeros(nset, rank))
        self.w_basis = nn.Parameter(torch.zeros(rank, out_dim))
        _init_weight(self.w_coeff)
        _init_weight(self.w_basis)

        self.b = nn.Parameter(torch.zeros(1, nset, c))

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, bias=True),
                nn.Conv2d(c, c, kernel_size=5, padding=2, groups=max(c // 4, 1), bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, bias=True),
                nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True),
            )

        self.conv1 = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.conv21 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)

        interc = max(2, min(c, 32))
        if interc % 2 != 0:
            interc = max(2, interc - 1)
        self.local_att = nn.Sequential(
            nn.Conv2d(c, interc, kernel_size=3, padding=1, groups=1, bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, kernel_size=1),
        )
        gl_channels = max(int(c * global_ratio), 1)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, gl_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gl_channels, self.nset, kernel_size=1, bias=True),
        )
        self.conv211 = nn.Conv2d(c, self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(dw_ch // 2, c, kernel_size=1, bias=True)

        self.conv4 = nn.Conv2d(c, ffn_ch, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, bias=True)

        if dropblock_prob > 0:
            self.dropout1 = DropBlock2D(drop_prob=dropblock_prob, block_size=dropblock_size)
            self.dropout2 = DropBlock2D(drop_prob=dropblock_prob, block_size=dropblock_size)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.sg = SimpleGate()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

        self.static_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        self.mix_alpha = nn.Parameter(torch.tensor(alpha_init).float().clamp(0.0, 1.0))

    def _resolve_kernel_bank(self):
        w_full = torch.matmul(self.w_coeff, self.w_basis)  # [nset, out_dim]
        return w_full.unsqueeze(0)

    @staticmethod
    def _apply_topk(att, topk):
        if topk is None or topk <= 0 or topk >= att.size(1):
            return att
        B, N, HW = att.shape
        values, indices = torch.topk(att, topk, dim=1)
        mask = torch.zeros_like(att)
        mask.scatter_(1, indices, 1.0)
        att = att * mask
        return att

    @staticmethod
    def _kba(x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        sca = self.sca(x)
        x1 = self.conv11(x)

        # Attention computation with ablation switches
        if self.disable_local_attention:
            local_att = torch.zeros_like(self.local_att(x))
        else:
            local_att = self.local_att(x)

        if self.disable_global_attention:
            global_att = torch.zeros_like(local_att)
        else:
            global_att = self.global_branch(x).expand_as(local_att)

        att = local_att + global_att + self.conv211(x)
        att = att * self.attgamma + att

        B, _, H, W = att.shape
        att_flat = att.view(B, self.nset, -1)
        att_flat = self._apply_topk(att_flat, self.topk)
        att = att_flat.view(B, self.nset, H, W)

        uf_pre = self.conv1(x)
        uf = self.conv21(uf_pre)
        static_out = self.static_conv(uf)

        if self.disable_dynamic_conv:
            # Only static conv
            x = static_out * self.ga1 + uf
        elif self.disable_static_conv:
            # Only dynamic conv
            dynamic_kernel = self._resolve_kernel_bank()
            x_dyn = self._kba(uf, att, self.k, self.g, self.b, dynamic_kernel)
            x = x_dyn * self.ga1 + uf
        else:
            # Hybrid (default)
            dynamic_kernel = self._resolve_kernel_bank()
            x_dyn = self._kba(uf, att, self.k, self.g, self.b, dynamic_kernel)
            alpha = torch.sigmoid(self.mix_alpha)
            x = (alpha * x_dyn + (1 - alpha) * static_out) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class UNetBlock(nn.Module):
    """Standard U-Net convolutional block: two consecutive 3x3 convolutions
    each followed by batch normalization and ReLU activation.

    This is used as a baseline for ablation study to compare with DAB-Block.
    """

    def __init__(
        self,
        c: int,
        dw_expand: float = 2.0,
        ffn_expand: float = 2.0,
        nset: int = 32,
        kernel_size: int = 3,
        groups_per_channel: int = 4,
        lightweight: bool = False,
        dropblock_prob: float = 0.0,
        dropblock_size: int = 7,
        topk: int = 0,
        low_rank_ratio: float = 0.25,
        alpha_init: float = 0.5,
        global_ratio: float = 0.5,
        stage_id: int = 0,
        disable_local_attention: bool = False,
        disable_global_attention: bool = False,
        disable_dynamic_conv: bool = False,
        disable_static_conv: bool = False,
    ):
        """Accept same parameters as KBBlock for compatibility, but ignore most of them."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
