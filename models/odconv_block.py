"""ODConv-Baseline block: faithful Li et al. 2022 omni-dimensional dynamic conv.

Reference: "Omni-Dimensional Dynamic Convolution", ICLR 2022.

Four orthogonal attention dimensions multiplied with the base kernel:
  alpha_s : spatial attention      [B, k*k]
  alpha_c : input-channel att      [B, in_c]
  alpha_f : output-channel att     [B, out_c]
  alpha_w : kernel-wise att        [B, K]   (across K experts)

The dynamic kernel is:
  W_dyn[b] = sum_k alpha_w[b,k] * (alpha_s[b,:,:,:] o alpha_c[b,:,:,:,:] o alpha_f[b,:,:,:,:] o experts[k])

We follow the official PyTorch reference (https://github.com/OSVAI/ODConv).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .kb_utils import LayerNorm2d, SimpleGate
from .dropblock import DropBlock2D

__all__ = ["ODConvBlock"]


class _Attention4D(nn.Module):
    """Generates the four attention maps used by ODConv."""

    def __init__(self, c: int, kernel_size: int, K: int, reduction: int = 4, init_temperature: float = 30.0):
        super().__init__()
        self.K = K
        self.k = kernel_size
        self.c = c
        hidden = max(c // reduction, 4)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # Post-GAP tensor is [B, hidden, 1, 1] -- spatial=1x1 means any
        # batch/group norm fails (need >=2 values per channel) when a training
        # batch has size 1 (last batch of BrEaST n_train=201 with batch=4).
        # The ODConv attention head only needs FC+ReLU; drop normalisation.
        self.fc = nn.Conv2d(c, hidden, kernel_size=1, bias=True)
        self.bn = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.alpha_c = nn.Conv2d(hidden, c, kernel_size=1, bias=True)
        self.alpha_f = nn.Conv2d(hidden, c, kernel_size=1, bias=True)
        self.alpha_s = nn.Conv2d(hidden, kernel_size * kernel_size, kernel_size=1, bias=True)
        self.alpha_w = nn.Conv2d(hidden, K, kernel_size=1, bias=True)

        self.register_buffer("temperature", torch.tensor(float(init_temperature)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        z = self.relu(self.bn(self.fc(self.gap(x))))                 # [B, hidden, 1, 1]
        a_c = torch.sigmoid(self.alpha_c(z)).view(B, 1, self.c, 1, 1)
        a_f = torch.sigmoid(self.alpha_f(z)).view(B, self.c, 1, 1, 1)
        a_s = torch.sigmoid(self.alpha_s(z)).view(B, 1, 1, self.k, self.k)
        a_w = F.softmax(self.alpha_w(z) / self.temperature, dim=1).view(B, self.K)
        return a_c, a_f, a_s, a_w


class ODConvLayer(nn.Module):
    """Omni-dimensional dynamic conv (Li 2022). Standard (groups=1) conv."""

    def __init__(self, c: int, kernel_size: int = 3, K: int = 4, init_temperature: float = 30.0):
        super().__init__()
        self.K = K
        self.c = c
        self.k = kernel_size
        self.experts = nn.Parameter(torch.empty(K, c, c, kernel_size, kernel_size))
        for k in range(K):
            init.kaiming_uniform_(self.experts[k], a=math.sqrt(5))

        self.attn = _Attention4D(c, kernel_size, K, init_temperature=init_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        a_c, a_f, a_s, a_w = self.attn(x)                              # shapes per attention dim
        # Apply spatial / channel multipliers to expert kernels
        # experts: [K, C_out, C_in, k, k]
        # a_c: [B, 1, C_in, 1, 1]; a_f: [B, C_out, 1, 1, 1]; a_s: [B, 1, 1, k, k]
        weight = self.experts.unsqueeze(0)                             # [1, K, C, C, k, k]
        weight = weight * a_f.unsqueeze(1)                             # broadcast on K
        weight = weight * a_c.unsqueeze(1)
        weight = weight * a_s.unsqueeze(1)
        # Combine across K experts using a_w
        weight = (weight * a_w.view(B, self.K, 1, 1, 1, 1)).sum(1)     # [B, C, C, k, k]
        weight = weight.reshape(B * C, C, self.k, self.k)
        # Apply via group conv (batch as groups)
        x_reshaped = x.reshape(1, B * C, H, W)
        out = F.conv2d(x_reshaped, weight, padding=self.k // 2, groups=B)
        return out.view(B, C, H, W)


class ODConvBlock(nn.Module):
    """Drop-in replacement for KBBlock using ODConv (Li 2022)."""

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
        odconv_K: int = 4,
        odconv_temperature: float = 30.0,
    ):
        super().__init__()
        self.k = kernel_size
        self.c = c
        dw_ch = int(c * dw_expand)
        ffn_ch = int(ffn_expand * c)

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

        self.odconv = ODConvLayer(c, kernel_size=kernel_size, K=odconv_K,
                                  init_temperature=odconv_temperature)

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
        x_dyn = self.odconv(uf)
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
