import json
import os

import numpy as np
import torch

from PIL import Image


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f"cond_rendered"]]
        stats["Cond rendered"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        image_root = os.path.join(root, "renders_cond", instance)
        with open(os.path.join(image_root, "transforms.json")) as f:
            metadata = json.load(f)
        n_views = len(metadata["frames"])
        view = np.random.randint(n_views)
        metadata = metadata["frames"][view]

        image_path = os.path.join(image_root, metadata["file_path"])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [
            int(aug_center[0] - aug_hsize),
            int(aug_center[1] - aug_hsize),
            int(aug_center[0] + aug_hsize),
            int(aug_center[1] + aug_hsize),
        ]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack["cond"] = image

        return pack


class ImageAllConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[(metadata["rendered"]) & (metadata["cond_rendered"])]
        stats["All rendered"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        image_root = os.path.join(root, "renders_all", instance)
        with open(os.path.join(image_root, "transforms.json")) as f:
            metadata = json.load(f)
        n_views = len(metadata["frames"])
        view = np.random.randint(n_views)
        metadata = metadata["frames"][view]

        image_path = os.path.join(image_root, metadata["file_path"])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [
            int(aug_center[0] - aug_hsize),
            int(aug_center[1] - aug_hsize),
            int(aug_center[0] + aug_hsize),
            int(aug_center[1] + aug_hsize),
        ]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack["cond"] = image

        return pack


class FloraResampleMixin:
    def __init__(self, roots, *, flora_ratio: float = 0.0, flora_match_substring: str = "flora", **kwargs):
        self.flora_ratio = flora_ratio
        self.flora_match_substring = flora_match_substring
        # Do not pass flora_* args further down the chain
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        # Build a cached flora subset for potential resampling; do not alter instances
        self.flora_metadata = metadata[
            metadata["sha256"].astype(str).str.contains(self.flora_match_substring, na=False)
        ]
        stats[f"Flora candidates ({self.flora_match_substring})"] = len(self.flora_metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        # Potentially resample the instance to a flora sample BEFORE any downstream loading
        if (
            getattr(self, "flora_ratio", 0.0) > 0.0
            and hasattr(self, "flora_metadata")
            and len(self.flora_metadata) > 0
            and np.random.rand() < self.flora_ratio
        ):
            instance = np.random.choice(self.flora_metadata["sha256"])  # aligned across all sub-loaders
        return super().get_instance(root, instance)


class MultiImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, view_count=3, **kwargs):
        self.image_size = image_size
        self.view_count = view_count
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[(metadata["rendered"]) & (metadata["fixview_rendered"])]
        stats["Fixview rendered"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        image_root = os.path.join(root, "renders_fixview", instance)
        with open(os.path.join(image_root, "transforms.json")) as f:
            metadata = json.load(f)
        frames = metadata["frames"]
        frames = sorted(frames, key=lambda x: x["file_path"])
        views = frames[: self.view_count]
        multi_view_images = []
        for i in range(self.view_count):
            metadata = views[i]

            image_path = os.path.join(image_root, metadata["file_path"])
            image = Image.open(image_path)

            alpha = np.array(image.getchannel(3))
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_size_ratio = 1.2
            aug_hsize = hsize * aug_size_ratio
            aug_center_offset = [0, 0]
            aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
            aug_bbox = [
                int(aug_center[0] - aug_hsize),
                int(aug_center[1] - aug_hsize),
                int(aug_center[0] + aug_hsize),
                int(aug_center[1] + aug_hsize),
            ]
            image = image.crop(aug_bbox)

            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = image.getchannel(3)
            image = image.convert("RGB")
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            image = image * alpha.unsqueeze(0)
            multi_view_images.append(image)
        pack["cond"] = torch.stack(multi_view_images, dim=-1)
        return pack


class PrevImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata["sha256_prev"].notna()]
        stats["With img prev"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        image_root = os.path.join(root, "renders_prev", instance)
        with open(os.path.join(image_root, "transforms.json")) as f:
            metadata = json.load(f)
        n_views = len(metadata["frames"])
        view = np.random.randint(n_views)
        metadata = metadata["frames"][view]

        image_path = os.path.join(image_root, metadata["file_path"])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [
            int(aug_center[0] - aug_hsize),
            int(aug_center[1] - aug_hsize),
            int(aug_center[0] + aug_hsize),
            int(aug_center[1] + aug_hsize),
        ]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack["cond"] = image

        return pack


class PrevImageCondConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata["sha256_prev"].notna()]
        stats["With img cond prev"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        image_root = os.path.join(root, "renders_cond_prev", instance)
        with open(os.path.join(image_root, "transforms.json")) as f:
            metadata = json.load(f)
        n_views = len(metadata["frames"])
        view = np.random.randint(n_views)
        metadata = metadata["frames"][view]

        image_path = os.path.join(image_root, metadata["file_path"])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [
            int(aug_center[0] - aug_hsize),
            int(aug_center[1] - aug_hsize),
            int(aug_center[0] + aug_hsize),
            int(aug_center[1] + aug_hsize),
        ]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack["cond"] = image

        return pack
