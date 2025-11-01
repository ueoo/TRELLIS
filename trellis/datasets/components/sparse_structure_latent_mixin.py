import os

import numpy as np
import torch

from PIL import Image

from ...modules import sparse as sp


class SparseStructureLatentConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata["sha256_prev"].notna()]
        stats["With ss prev"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        # in ss_latents_prev, the name of the file is the sha256 of the previous instance
        latent = np.load(os.path.join(root, "ss_latents_prev", self.latent_model, f"{instance}.npz"))
        z = torch.tensor(latent["mean"]).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack["cond"] = z

        return pack
