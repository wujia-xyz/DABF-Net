"""
EMCAD-inspired decoder with KBBlock-based MSCAM replacements.
"""
from functools import partial
from typing import List, Optional
import torch
import torch.nn as nn

from .kb_blocks import KBBlock, UNetBlock
from .kba_baseline_block import KBABaselineBlock
from .dyconv_block import DyConvBlock
from .condconv_block import CondConvBlock
from .odconv_block import ODConvBlock

__all__ = ["HybridEMCADDecoder"]


def _init_weights(m: nn.Module, scheme: str = "kaiming") -> None:
    if isinstance(m, nn.Conv2d):
        if scheme == "kaiming":
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


class BilinearUp(nn.Module):
    """Simple bilinear upsampling as alternative to PixelShuffle."""

    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.apply(partial(_init_weights, scheme="kaiming"))

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class PixelShuffleUp(nn.Module):
    """
    Two-stage upsampling:
        1) 1x1 conv + PixelShuffle to enlarge resolution without interpolation artifacts
        2) depth-wise refinement (optionally followed by channel attention)
    """

    def __init__(self, in_channels, out_channels, scale=2, use_attention=False):
        super().__init__()
        self.scale = scale
        self.expand = nn.Conv2d(in_channels, in_channels * (scale ** 2), kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.dw_refine = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.se = None
        self.apply(partial(_init_weights, scheme="kaiming"))

    def forward(self, x):
        x = self.expand(x)
        x = self.pixel_shuffle(x)
        x = self.dw_refine(x)
        x = self.proj(x)
        if self.se is not None:
            x = x * self.se(x)
        return x


class LGAG(nn.Module):
    """Enhanced LGAG with multi-scale conv + channel attention + residual gating."""

    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, use_residual=True):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        padding = kernel_size // 2

        self.g_large = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.g_small = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, padding=1, groups=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.x_large = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.x_small = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, padding=1, groups=1, bias=True),
            nn.BatchNorm2d(F_int),
        )

        se_channels = max(F_int // 4, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * F_int, se_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, 2 * F_int, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(F_int),
            nn.GELU(),
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.use_residual = use_residual
        self.lambda_res = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, g, x):
        g_feat = self.g_large(g) + self.g_small(g)
        x_feat = self.x_large(x) + self.x_small(x)

        cat = torch.cat([g_feat, x_feat], dim=1)
        gate = self.channel_gate(cat)
        g_gate, x_gate = torch.chunk(gate, 2, dim=1)
        g_feat = g_feat * g_gate
        x_feat = x_feat * x_gate

        merged = g_feat + x_feat
        psi = self.spatial_gate(merged)

        out = x * psi
        if self.use_residual:
            out = out + torch.sigmoid(self.lambda_res) * x
        return out


class HybridEMCADDecoder(nn.Module):
    """EMCAD decoder where MSCAM modules are replaced by KBBlocks."""

    def __init__(
        self,
        channels: List[int],
        lgag_kernel: int = 3,
        decoder_nsets: Optional[List[int]] = None,
        decoder_topk: Optional[List[int]] = None,
        kba_low_rank_ratio: float = 0.25,
        kba_alpha_init: float = 0.5,
        kba_global_ratio: float = 0.5,
        kbblock_kwargs: Optional[dict] = None,
        use_lgag: bool = True,
        use_pixelshuffle: bool = True,
        disable_local_attention: bool = False,
        disable_global_attention: bool = False,
        disable_dynamic_conv: bool = False,
        disable_static_conv: bool = False,
        use_unet_block: bool = False,
        use_kba_baseline: bool = False,
        use_dyconv: bool = False,
        use_condconv: bool = False,
        use_odconv: bool = False,
    ):
        super().__init__()
        kbblock_kwargs = kbblock_kwargs or {}
        self.decoder_nsets = decoder_nsets
        self.decoder_topk = decoder_topk
        self.kba_low_rank_ratio = kba_low_rank_ratio
        self.kba_alpha_init = kba_alpha_init
        self.kba_global_ratio = kba_global_ratio
        self.stage_counter = 0
        self.use_lgag = use_lgag and lgag_kernel > 0
        self.use_pixelshuffle = use_pixelshuffle
        self.disable_local_attention = disable_local_attention
        self.disable_global_attention = disable_global_attention
        self.disable_dynamic_conv = disable_dynamic_conv
        self.disable_static_conv = disable_static_conv
        self.use_unet_block = use_unet_block
        self.use_kba_baseline = use_kba_baseline
        self.use_dyconv = use_dyconv
        self.use_condconv = use_condconv
        self.use_odconv = use_odconv

        # Select block type. Priority: dyconv -> condconv -> odconv -> kba_baseline
        # -> unet_block -> KBBlock (default DAB-Block)
        if use_dyconv:
            self.BlockClass = DyConvBlock
        elif use_condconv:
            self.BlockClass = CondConvBlock
        elif use_odconv:
            self.BlockClass = ODConvBlock
        elif use_kba_baseline:
            self.BlockClass = KBABaselineBlock
        elif use_unet_block:
            self.BlockClass = UNetBlock
        else:
            self.BlockClass = KBBlock

        self.stage4 = self._make_block(channels[0], kbblock_kwargs)

        # Choose upsampling method based on use_pixelshuffle flag
        UpModule = PixelShuffleUp if use_pixelshuffle else BilinearUp

        self.eucb3 = UpModule(channels[0], channels[1])
        if self.use_lgag:
            self.lgag3 = LGAG(
                F_g=channels[1],
                F_l=channels[1],
                F_int=channels[1] // 2,
                kernel_size=lgag_kernel,
                groups=max(1, channels[1] // 2),
            )
        else:
            self.lgag3 = None
        self.stage3 = self._make_block(channels[1], kbblock_kwargs)
        self.refine3 = self._make_block(channels[1], kbblock_kwargs)

        self.eucb2 = UpModule(channels[1], channels[2])
        if self.use_lgag:
            self.lgag2 = LGAG(
                F_g=channels[2],
                F_l=channels[2],
                F_int=channels[2] // 2,
                kernel_size=lgag_kernel,
                groups=max(1, channels[2] // 2),
            )
        else:
            self.lgag2 = None
        self.stage2 = self._make_block(channels[2], kbblock_kwargs)
        self.refine2 = self._make_block(channels[2], kbblock_kwargs)

        self.eucb1 = UpModule(channels[2], channels[3])
        if self.use_lgag:
            self.lgag1 = LGAG(
                F_g=channels[3],
                F_l=channels[3],
                F_int=max(1, channels[3] // 2),
                kernel_size=lgag_kernel,
                groups=max(1, channels[3] // 2),
            )
        else:
            self.lgag1 = None
        self.stage1 = self._make_block(channels[3], kbblock_kwargs)
        self.refine1 = self._make_block(channels[3], kbblock_kwargs)

    @staticmethod
    def _stage_value(schedule, idx, default):
        if schedule is None:
            return default
        if isinstance(schedule, (list, tuple)):
            if idx < len(schedule):
                return schedule[idx]
            return schedule[-1]
        return schedule

    def _make_block(self, channel, kbblock_kwargs):
        idx = self.stage_counter
        self.stage_counter += 1
        return self.BlockClass(
            channel,
            nset=self._stage_value(self.decoder_nsets, idx, 32),
            topk=self._stage_value(self.decoder_topk, idx, 0),
            low_rank_ratio=self.kba_low_rank_ratio,
            alpha_init=self.kba_alpha_init,
            global_ratio=self.kba_global_ratio,
            stage_id=idx,
            disable_local_attention=self.disable_local_attention,
            disable_global_attention=self.disable_global_attention,
            disable_dynamic_conv=self.disable_dynamic_conv,
            disable_static_conv=self.disable_static_conv,
            **kbblock_kwargs,
        )

    def forward(self, x, skips: List[torch.Tensor]):
        assert len(skips) >= 3, "Hybrid decoder expects at least 3 skip connections."

        d4 = self.stage4(x)

        d3 = self.eucb3(d4)
        if self.lgag3 is not None:
            d3 = d3 + self.lgag3(d3, skips[0])
        else:
            d3 = d3 + skips[0]  # Simple addition when LGAG disabled
        d3 = self.stage3(d3)
        d3 = self.refine3(d3)

        d2 = self.eucb2(d3)
        if self.lgag2 is not None:
            d2 = d2 + self.lgag2(d2, skips[1])
        else:
            d2 = d2 + skips[1]
        d2 = self.stage2(d2)
        d2 = self.refine2(d2)

        d1 = self.eucb1(d2)
        if self.lgag1 is not None:
            d1 = d1 + self.lgag1(d1, skips[2])
        else:
            d1 = d1 + skips[2]
        d1 = self.stage1(d1)
        d1 = self.refine1(d1)

        return [d4, d3, d2, d1]
