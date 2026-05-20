"""
KBNet encoder reused for the hybrid model.
"""
from typing import List, Sequence, Optional
import torch
import torch.nn as nn

from .kb_blocks import KBBlock, UNetBlock
from .kba_baseline_block import KBABaselineBlock
from .dyconv_block import DyConvBlock
from .condconv_block import CondConvBlock
from .odconv_block import ODConvBlock

__all__ = ["KBNetEncoder"]


class KBNetEncoder(nn.Module):
    """Encoder path from KBNet, exposing multi-scale feature maps."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 64,
        enc_blk_nums: Sequence[int] = (2, 2, 4, 8),
        middle_blk_num: int = 12,
        ffn_scale: float = 2.0,
        lightweight: bool = False,
        dropblock_prob: float = 0.0,
        dropblock_size: int = 7,
        encoder_nsets: Optional[Sequence[int]] = None,
        encoder_topk: Optional[Sequence[int]] = None,
        kba_low_rank_ratio: float = 0.25,
        kba_alpha_init: float = 0.5,
        kba_global_ratio: float = 0.5,
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
            BlockClass = DyConvBlock
        elif use_condconv:
            BlockClass = CondConvBlock
        elif use_odconv:
            BlockClass = ODConvBlock
        elif use_kba_baseline:
            BlockClass = KBABaselineBlock
        elif use_unet_block:
            BlockClass = UNetBlock
        else:
            BlockClass = KBBlock

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        self.stage_channels: List[int] = []
        for stage_idx, num in enumerate(enc_blk_nums):
            self.stage_channels.append(chan)
            blocks = [
                BlockClass(
                    chan,
                    ffn_expand=ffn_scale,
                    lightweight=lightweight,
                    dropblock_prob=dropblock_prob,
                    dropblock_size=dropblock_size,
                    nset=self._stage_value(encoder_nsets, stage_idx, 32),
                    topk=self._stage_value(encoder_topk, stage_idx, 0),
                    low_rank_ratio=kba_low_rank_ratio,
                    alpha_init=kba_alpha_init,
                    global_ratio=kba_global_ratio,
                    stage_id=stage_idx,
                    disable_local_attention=disable_local_attention,
                    disable_global_attention=disable_global_attention,
                    disable_dynamic_conv=disable_dynamic_conv,
                    disable_static_conv=disable_static_conv,
                )
                for _ in range(num)
            ]
            self.encoders.append(nn.Sequential(*blocks))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        self.bottleneck_channels = chan
        mid_stage_idx = len(enc_blk_nums)
        self.middle_blks = nn.Sequential(
            *[
                BlockClass(
                    chan,
                    ffn_expand=ffn_scale,
                    lightweight=lightweight,
                    dropblock_prob=dropblock_prob,
                    dropblock_size=dropblock_size,
                    nset=self._stage_value(encoder_nsets, mid_stage_idx, 32),
                    topk=self._stage_value(encoder_topk, mid_stage_idx, 0),
                    low_rank_ratio=kba_low_rank_ratio,
                    alpha_init=kba_alpha_init,
                    global_ratio=kba_global_ratio,
                    stage_id=mid_stage_idx,
                    disable_local_attention=disable_local_attention,
                    disable_global_attention=disable_global_attention,
                    disable_dynamic_conv=disable_dynamic_conv,
                    disable_static_conv=disable_static_conv,
                )
                for _ in range(middle_blk_num)
            ]
        )

    @staticmethod
    def _stage_value(schedule, idx, default):
        if schedule is None:
            return default
        if isinstance(schedule, (list, tuple)):
            if idx < len(schedule):
                return schedule[idx]
            return schedule[-1]
        return schedule

    def forward(self, x: torch.Tensor):
        feats = []
        x = self.intro(x)

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            feats.append(x)
            x = down(x)

        x = self.middle_blks(x)
        return x, feats
