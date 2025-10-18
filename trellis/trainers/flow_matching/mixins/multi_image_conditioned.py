from typing import *

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from ....utils import dist_utils
from .image_conditioned import ImageConditionedMixin


class MultiImageConditionedMixin(ImageConditionedMixin):
    """
    Mixin for multi-image conditioning.

    Updated dataset contract:
    - Dataset now sets `pack["cond"]` to shape [3, H, W, V] per sample.
      After collation: [B, 3, H, W, V].
    - We flatten views to the batch, encode once with DINOv2, then reshape
      back and concatenate tokens across views along the sequence dimension.
    """

    def __init__(self, *args, view_count: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Optional hint; if None we infer from kwargs at call time.
        self.view_count = view_count

    @torch.no_grad()
    def encode_multi_image(self, cond: Optional[torch.Tensor], **kwargs) -> torch.Tensor:
        # Expect new contract: cond is [B, C, H, W, V]
        assert isinstance(cond, torch.Tensor) and cond.ndim == 5, "cond must be a 5D tensor [B,C,H,W,V]"
        B, C, H, W, V = cond.shape
        if self.image_cond_model is None:
            self._init_image_cond_model()
        # Flatten views into batch: [B*V, C, H, W]
        imgs = cond.permute(0, 4, 1, 2, 3).reshape(B * V, C, H, W)
        imgs = F.interpolate(imgs, size=(518, 518), mode="bilinear", align_corners=False)
        imgs = self.image_cond_model["transform"](imgs).cuda()
        feats = self.image_cond_model["model"](imgs, is_training=True)["x_prenorm"]  # [B*V, N, D]
        tokens = F.layer_norm(feats, feats.shape[-1:])
        # Concatenate views along token dimension (dim=-2)
        N, D = tokens.shape[-2], tokens.shape[-1]
        tokens = tokens.view(B, V, N, D).reshape(B, V * N, D)
        return tokens

    def get_cond(self, cond, **kwargs):
        cond = self.encode_multi_image(cond, **kwargs)
        kwargs["neg_cond"] = torch.zeros_like(cond)
        # Bypass ImageConditionedMixin.get_cond to avoid re-encoding as images
        cond = super(ImageConditionedMixin, self).get_cond(cond, **kwargs)
        return cond

    def get_inference_cond(self, cond, **kwargs):
        cond = self.encode_multi_image(cond, **kwargs)
        kwargs["neg_cond"] = torch.zeros_like(cond)
        # Bypass ImageConditionedMixin.get_inference_cond to avoid re-encoding as images
        cond = super(ImageConditionedMixin, self).get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        if isinstance(cond, torch.Tensor) and cond.ndim == 5:
            img = cond[..., 0]
            return {"image": {"value": img, "type": "image"}}
        return {}
