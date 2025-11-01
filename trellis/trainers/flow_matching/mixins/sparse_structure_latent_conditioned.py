from typing import *

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from ....utils import dist_utils


class SparseStructureLatentConditionedMixin:
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
        if hasattr(
            super(SparseStructureLatentConditionedMixin, SparseStructureLatentConditionedMixin), "prepare_for_training"
        ):
            super(SparseStructureLatentConditionedMixin, SparseStructureLatentConditionedMixin).prepare_for_training(
                **kwargs
            )

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        kwargs["neg_cond"] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        kwargs["neg_cond"] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond
