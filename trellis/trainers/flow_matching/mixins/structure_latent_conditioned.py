from typing import *

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from ....modules import sparse as sp
from ....utils import dist_utils


class SLatConditionedMixin:
    """
    Mixin for image-conditioned models.

    Args:
        image_cond_model: The image conditioning model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def prepare_for_training(**kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(SLatConditionedMixin, SLatConditionedMixin), "prepare_for_training"):
            super(SLatConditionedMixin, SLatConditionedMixin).prepare_for_training(**kwargs)

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        kwargs["neg_cond"] = sp.SparseTensor(coords=cond.coords, feats=torch.zeros_like(cond.feats))
        cond = super().get_cond(cond, **kwargs)
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        kwargs["neg_cond"] = sp.SparseTensor(coords=cond.coords, feats=torch.zeros_like(cond.feats))
        cond = super().get_inference_cond(cond, **kwargs)
        return cond
