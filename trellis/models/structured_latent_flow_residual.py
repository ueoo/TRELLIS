from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import sparse as sp
from ..modules.norm import LayerNorm32
from ..modules.sparse.transformer import (
    ModulatedSparseTransformerBlock,
    ModulatedSparseTransformerResidualBlock,
)
from ..modules.sparse.transformer.blocks import SparseFeedForwardNet
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.transformer.blocks import FeedForwardNet
from ..modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module
from .sparse_elastic_mixin import SparseTransformerElasticMixin
from .sparse_structure_flow import TimestepEmbedder
from .structured_latent_flow import SparseResBlock3d


class SLatResidualSLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        residual_position: Literal["every", "final"] = "final",
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.residual_position = residual_position

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(
                io_block_channels
            ), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(
            in_channels, model_channels if io_block_channels is None else io_block_channels[0]
        )
        self.input_layer_cond = sp.SparseLinear(in_channels, cond_channels)

        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                self.input_blocks.extend(
                    [
                        SparseResBlock3d(
                            chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks - 1)
                    ]
                )
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )

        if self.residual_position == "every":
            self.blocks = nn.ModuleList(
                [
                    ModulatedSparseTransformerResidualBlock(
                        model_channels,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        attn_mode="full",
                        use_checkpoint=self.use_checkpoint,
                        use_rope=(pe_mode == "rope"),
                        share_mod=self.share_mod,
                        qk_rms_norm=self.qk_rms_norm,
                    )
                    for _ in range(num_blocks)
                ]
            )
        elif self.residual_position == "final":
            self.blocks = nn.ModuleList(
                [
                    ModulatedSparseTransformerBlock(
                        model_channels,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        attn_mode="full",
                        use_checkpoint=self.use_checkpoint,
                        use_rope=(pe_mode == "rope"),
                        share_mod=self.share_mod,
                        qk_rms_norm=self.qk_rms_norm,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(
                reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))
            ):
                self.out_blocks.append(
                    SparseResBlock3d(
                        prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels,
                        out_channels=chs,
                        upsample=True,
                    )
                )
                self.out_blocks.extend(
                    [
                        SparseResBlock3d(
                            chs * 2 if self.use_skip_connection else chs,
                            model_channels,
                            out_channels=chs,
                        )
                        for _ in range(num_io_res_blocks - 1)
                    ]
                )

        self.out_layer = sp.SparseLinear(
            model_channels if io_block_channels is None else io_block_channels[0], out_channels
        )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)

        cond = self.input_layer_cond(cond)
        cond = cond.type(self.dtype)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
            cond = cond + self.pos_embedder(cond.coords[:, 1:]).type(self.dtype)

        # Align cond's spatial resolution with h before residual blocks when needed
        cond_context = cond
        if self.residual_position == "every" and self.io_block_channels is not None:
            for _ in range(len(self.io_block_channels)):
                cond_context = sp.SparseDownsample(2)(cond_context)

        if self.residual_position == "every":
            for block in self.blocks:
                h = block(h, t_emb, cond_context)
        elif self.residual_position == "final":
            for block in self.blocks:
                h = block(h, t_emb)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        # Align cond to h for safe addition in case coords differ
        cond_aligned = sp.align_to(h, cond)
        h = h + cond_aligned
        h = self.out_layer(h.type(x.dtype))
        return h


class ElasticSLatResidualSLatFlowModel(SparseTransformerElasticMixin, SLatResidualSLatFlowModel):
    """
    SLat Cond SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """

    pass
