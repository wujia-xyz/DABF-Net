"""DyConv-Baseline block: faithful Chen et al. 2020 dynamic convolution.

Reference: "Dynamic Convolution: Attention over Convolution Kernels", CVPR 2020.

Replaces DAB-Block's KBA-based dynamic mechanism with a K-expert weighted-kernel
formulation:
    att = SoftMax(GAP(x) -> FC -> ReLU -> FC, dim=K, temp=τ)         # [B, K]
    weight = sum_k att[k] * experts[k]                                # [B, C, C, k, k]
    out = F.conv2d(x, weight)  (per-batch; implemented via grouped conv)

Outer block shell (norm, sca, conv11, FFN, dropouts, residuals, beta/gamma) is
identical to KBBlock so the comparison isolates the dynamic-conv mechanism.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .kb_utils import LayerNorm2d, SimpleGate
from .dropblock import DropBlock2D

__all__ = ["DyConvBlock"]


class DyConvLayer(nn.Module):
    """K-expert dynamic conv (Chen 2020). Standard (groups=1) full conv."""

    def __init__(self, c: int, kernel_size: int = 3, K: int = 4, reduction: int = 4, init_temperature: float = 30.0):
        super().__init__()
        self.K = K
        self.c = c
        self.k = kernel_size
        self.experts = nn.Parameter(torch.empty(K, c, c, kernel_size, kernel_size))
        for k in range(K):
            init.kaiming_uniform_(self.experts[k], a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(K, c))

        hidden = max(c // reduction, 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, K, kernel_size=1, bias=True),
        )
        # Temperature anneals from init -> 1 across training (handled externally if desired)
        self.register_buffer("temperature", torch.tensor(float(init_temperature)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        att = self.attention(x).view(B, self.K)
        att = F.softmax(att / self.temperature, dim=1)                    # [B, K]
        # Combine experts to per-batch kernel
        weight = (att.view(B, self.K, 1, 1, 1, 1) *
                  self.experts.unsqueeze(0)).sum(1).reshape(B * C, C, self.k, self.k)
        bias = (att @ self.bias).view(B * C)
        # Apply via group conv (batch-as-groups trick)
        x_reshaped = x.reshape(1, B * C, H, W)
        out = F.conv2d(x_reshaped, weight, bias=bias,
                       padding=self.k // 2, groups=B)
        return out.view(B, C, H, W)


class DyConvBlock(nn.Module):
    """Drop-in replacement for KBBlock using DyConv (Chen 2020) dynamic mechanism."""

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
        # Accepted for signature compatibility; unused
        topk: int = 0,
        low_rank_ratio: float = 0.25,
        alpha_init: float = 0.5,
        global_ratio: float = 0.5,
        stage_id: int = 0,
        disable_local_attention: bool = False,
        disable_global_attention: bool = False,
        disable_dynamic_conv: bool = False,
        disable_static_conv: bool = False,
        # DyConv specific
        dyconv_K: int = 4,
        dyconv_temperature: float = 30.0,
    ):
        super().__init__()
        self.k = kernel_size
        self.c = c
        dw_ch = int(c * dw_expand)
        ffn_ch = int(ffn_expand * c)

        # Outer shell identical to KBABaselineBlock / KBBlock
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

        # Replace DAB inner mechanism with DyConv layer
        self.dyconv = DyConvLayer(c, kernel_size=kernel_size, K=dyconv_K, init_temperature=dyconv_temperature)

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
        self.sg = SimpleGate()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        sca = self.sca(x)
        x1 = self.conv11(x)

        uf_pre = self.conv1(x)
        uf = self.conv21(uf_pre)
        x_dyn = self.dyconv(uf)
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
