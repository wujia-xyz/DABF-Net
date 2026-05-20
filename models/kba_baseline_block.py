"""KBA-Baseline block: faithful KBNet original KBA, drop-in for KBBlock (DAB-Block).

This block reproduces the attention + dynamic-convolution mechanism from KBNet's
MFF (basicsr/models/archs/kbnet_l_arch.py::MFF + kb_utils.py::KBAFunction).
All four DAB-specific innovations are removed:
    1. multi-branch attention   -> single-branch (conv2 + conv211)
    2. low-rank factorization   -> full-rank kernel bank (1, nset, out_dim)
    3. top-k sparsification     -> none
    4. hybrid static-dynamic    -> pure dynamic
The outer block structure (norm, SCA, conv11, FFN, DropBlock, beta/gamma
residual scaling) is kept identical to KBBlock so the contribution of the four
DAB-specific innovations is isolated.

Used to construct the DABF-Net "KBA-Baseline" ablation that replaces every
KBBlock in the encoder/decoder with this KBABaselineBlock. Address Reviewer 1
comment 4 (R1.4): "How does a KBNet-based U-Net baseline (replacing DAB-Block
with KBNet's original KBA module) perform?"
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from .kb_utils import KBAFunction, LayerNorm2d, SimpleGate
from .dropblock import DropBlock2D

__all__ = ["KBABaselineBlock"]


class KBABaselineBlock(nn.Module):
    """Drop-in replacement for KBBlock using KBNet's original KBA mechanism."""

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
        # Accepted for signature compatibility with KBBlock; intentionally unused.
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

        dw_ch = int(c * dw_expand)
        ffn_ch = int(ffn_expand * c)
        self.g = max(1, c // max(groups_per_channel, 1))
        out_dim = c * c // self.g * self.k ** 2

        # ---- ORIGINAL-KBA full-rank kernel bank (matches KBNet self.w) ----
        self.w = nn.Parameter(torch.zeros(1, nset, out_dim))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        init.kaiming_uniform_(self.w, a=math.sqrt(5))

        # ---- shell identical to KBBlock ----
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

        # ---- SINGLE-branch attention, mirroring MFF ----
        interc = max(2, min(c, 32))
        if interc % 2 != 0:
            interc = max(2, interc - 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, interc, kernel_size=3, padding=1, groups=1, bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, kernel_size=1),
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

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        sca = self.sca(x)
        x1 = self.conv11(x)

        # Single-branch attention (NO global, NO top-k)
        att = self.conv2(x) * self.attgamma + self.conv211(x)

        uf_pre = self.conv1(x)
        uf = self.conv21(uf_pre)

        # Pure dynamic KBA, full-rank kernel bank, NO static blending
        x_dyn = KBAFunction.apply(uf, att, self.k, self.g, self.b, self.w)
        x = x_dyn * self.ga1 + uf

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
