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
from .structured_latent_flow import SLatFlowModel, SparseResBlock3d


class ControlSLatCondSLatFlowModel(SLatFlowModel):
    def __init__(self, *args, downsample_cond_for_control: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsample_cond_for_control = downsample_cond_for_control
        # Cond path projects to cond_channels first, then to model_channels for control
        self.input_layer_cond = sp.SparseLinear(self.in_channels, self.cond_channels)
        if self.mlp_cond:
            self.mlp_layer_cond = SparseFeedForwardNet(self.cond_channels, mlp_ratio=self.mlp_ratio)

        # Map cond features to model dimension used by the core blocks
        if self.cond_channels != self.model_channels:
            self.cond_to_model = sp.SparseLinear(self.cond_channels, self.model_channels)
        else:
            self.cond_to_model = nn.Identity()

        # Replace cross-attn blocks with self-attn blocks and add zero-inited control projections per block
        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerBlock(
                    self.model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(self.pe_mode == "rope"),
                    share_mod=self.share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Zero-initialized per-block control injection layers (ControlNet-style)
        self.control_projs = nn.ModuleList(
            [zero_module(sp.SparseLinear(self.model_channels, self.model_channels)) for _ in range(self.num_blocks)]
        )

        # Ensure dtype consistency when fp16 is enabled
        if self.use_fp16:
            self.blocks.apply(convert_module_to_f16)
            self.control_projs.apply(convert_module_to_f16)

    def convert_to_fp16(self) -> None:
        super().convert_to_fp16()
        if hasattr(self, "blocks"):
            self.blocks.apply(convert_module_to_f16)
        if hasattr(self, "control_projs"):
            self.control_projs.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        super().convert_to_fp32()
        if hasattr(self, "blocks"):
            self.blocks.apply(convert_module_to_f32)
        if hasattr(self, "control_projs"):
            self.control_projs.apply(convert_module_to_f32)

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

        # Prepare cond features in model dimension (keep in fp32 path like original cond flow)
        cond_f = self.input_layer_cond(cond)
        if self.mlp_cond:
            cond_f = self.mlp_layer_cond(cond_f)
        cond_f = self.cond_to_model(cond_f)
        cond_f = cond_f.type(self.dtype)

        # Positional embeddings if needed
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
            cond_f = cond_f + self.pos_embedder(cond_f.coords[:, 1:]).type(self.dtype)

        # Downsample cond to match transformer stage spatial resolution
        cond_context = cond_f
        if self.downsample_cond_for_control and self.io_block_channels is not None:
            for _ in range(len(self.io_block_channels)):
                cond_context = sp.SparseDownsample(2)(cond_context)

        # Core transformer blocks with additive control injection
        for i, block in enumerate(self.blocks):
            h = block(h, t_emb)
            # Align cond coords to h before injection, then project and cast to dtype
            cond_aligned = sp.align_to(h, cond_context)
            ctrl = self.control_projs[i](cond_aligned)
            h = h + ctrl.type(self.dtype)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h


class ElasticControlSLatCondSLatFlowModel(SparseTransformerElasticMixin, ControlSLatCondSLatFlowModel):
    """
    ControlNet SLat Cond SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """

    pass
