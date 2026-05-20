"""
Hybrid model: KBNet encoder + EMCAD decoder (MSCAM -> KBBlock).
"""
from typing import List, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kbnet_encoder import KBNetEncoder
from .hybrid_decoder import HybridEMCADDecoder

__all__ = ["KBNetEMCADHybrid", "build_hybrid_model"]


class KBNetEMCADHybrid(nn.Module):
    def __init__(
        self,
        img_channel: int = 3,
        num_classes: int = 1,
        width: int = 64,
        enc_blk_nums: Sequence[int] = (2, 2, 4, 8),
        middle_blk_num: int = 12,
        ffn_scale: float = 2.0,
        lightweight: bool = False,
        dropblock_prob: float = 0.0,
        dropblock_size: int = 7,
        lgag_kernel: int = 3,
        encoder_nsets: Optional[Sequence[int]] = None,
        decoder_nsets: Optional[Sequence[int]] = None,
        encoder_topk: Optional[Sequence[int]] = None,
        decoder_topk: Optional[Sequence[int]] = None,
        kba_low_rank_ratio: float = 0.25,
        kba_alpha_init: float = 0.5,
        kba_global_ratio: float = 0.5,
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

        self.encoder = KBNetEncoder(
            img_channel=img_channel,
            width=width,
            enc_blk_nums=enc_blk_nums,
            middle_blk_num=middle_blk_num,
            ffn_scale=ffn_scale,
            lightweight=lightweight,
            dropblock_prob=dropblock_prob,
            dropblock_size=dropblock_size,
            encoder_nsets=encoder_nsets,
            encoder_topk=encoder_topk,
            kba_low_rank_ratio=kba_low_rank_ratio,
            kba_alpha_init=kba_alpha_init,
            kba_global_ratio=kba_global_ratio,
            disable_local_attention=disable_local_attention,
            disable_global_attention=disable_global_attention,
            disable_dynamic_conv=disable_dynamic_conv,
            disable_static_conv=disable_static_conv,
            use_unet_block=use_unet_block,
            use_kba_baseline=use_kba_baseline,
            use_dyconv=use_dyconv,
            use_condconv=use_condconv,
            use_odconv=use_odconv,
        )

        if len(self.encoder.stage_channels) < 3:
            raise ValueError("Encoder must provide at least 3 skip stages for the decoder.")

        decoder_channels: List[int] = [
            self.encoder.bottleneck_channels,
            self.encoder.stage_channels[-1],
            self.encoder.stage_channels[-2],
            self.encoder.stage_channels[-3],
        ]

        decoder_block_kwargs = dict(
            dw_expand=2.0,
            ffn_expand=ffn_scale,
            lightweight=lightweight,
            dropblock_prob=dropblock_prob,
            dropblock_size=dropblock_size,
        )

        self.decoder = HybridEMCADDecoder(
            channels=decoder_channels,
            lgag_kernel=lgag_kernel,
            decoder_nsets=decoder_nsets,
            decoder_topk=decoder_topk,
            kba_low_rank_ratio=kba_low_rank_ratio,
            kba_alpha_init=kba_alpha_init,
            kba_global_ratio=kba_global_ratio,
            kbblock_kwargs=decoder_block_kwargs,
            use_lgag=use_lgag,
            use_pixelshuffle=use_pixelshuffle,
            disable_local_attention=disable_local_attention,
            disable_global_attention=disable_global_attention,
            disable_dynamic_conv=disable_dynamic_conv,
            disable_static_conv=disable_static_conv,
            use_unet_block=use_unet_block,
            use_kba_baseline=use_kba_baseline,
            use_dyconv=use_dyconv,
            use_condconv=use_condconv,
            use_odconv=use_odconv,
        )

        self.out_head4 = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)
        self.out_head3 = nn.Conv2d(decoder_channels[1], num_classes, kernel_size=1)
        self.out_head2 = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)
        self.out_head1 = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

        self.num_classes = num_classes
        self.padder_size = 2 ** len(self.encoder.stage_channels)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, H, W = inp.shape
        x = self.check_image_size(inp)
        _, _, Hp, Wp = x.shape
        bottleneck, feats = self.encoder(x)

        skips = [feats[-1], feats[-2], feats[-3]]
        dec_outs = self.decoder(bottleneck, skips)

        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        preds = []
        for p in [p4, p3, p2, p1]:
            up = F.interpolate(p, size=(Hp, Wp), mode="bilinear", align_corners=False)
            preds.append(up[:, :, :H, :W])

        return preds

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def build_hybrid_model(config: dict) -> KBNetEMCADHybrid:
    """Factory used by training scripts."""
    model = KBNetEMCADHybrid(
        img_channel=config.get("img_channel", 3),
        num_classes=config.get("num_classes", 1),
        width=config.get("width", 64),
        enc_blk_nums=config.get("enc_blk_nums", (2, 2, 4, 8)),
        middle_blk_num=config.get("middle_blk_num", 12),
        ffn_scale=config.get("ffn_scale", 2.0),
        lightweight=config.get("lightweight", False),
        dropblock_prob=config.get("dropblock_prob", 0.0),
        dropblock_size=config.get("dropblock_size", 7),
        lgag_kernel=config.get("lgag_kernel", 3),
        encoder_nsets=config.get("encoder_nsets"),
        decoder_nsets=config.get("decoder_nsets"),
        encoder_topk=config.get("encoder_topk"),
        decoder_topk=config.get("decoder_topk"),
        kba_low_rank_ratio=config.get("kba_low_rank_ratio", 0.25),
        kba_alpha_init=config.get("kba_alpha_init", 0.5),
        kba_global_ratio=config.get("kba_global_ratio", 0.5),
        use_lgag=config.get("use_lgag", True),
        use_pixelshuffle=config.get("use_pixelshuffle", True),
        disable_local_attention=config.get("disable_local_attention", False),
        disable_global_attention=config.get("disable_global_attention", False),
        disable_dynamic_conv=config.get("disable_dynamic_conv", False),
        disable_static_conv=config.get("disable_static_conv", False),
        use_unet_block=config.get("use_unet_block", False),
        use_kba_baseline=config.get("use_kba_baseline", False),
        use_dyconv=config.get("use_dyconv", False),
        use_condconv=config.get("use_condconv", False),
        use_odconv=config.get("use_odconv", False),
    )
    return model
