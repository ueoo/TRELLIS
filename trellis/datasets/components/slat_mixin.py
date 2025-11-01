import os

import numpy as np
import torch

from ...modules import sparse as sp


class SLatConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f"sha256_prev"].notna()]
        stats["With ss prev"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        data = np.load(os.path.join(root, "latents_prev", self.latent_model, f"{instance}.npz"))
        coords = torch.tensor(data["coords"]).int()
        feats = torch.tensor(data["feats"]).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        cond = sp.SparseTensor(coords=coords, feats=feats)
        pack["cond"] = cond
        return pack
