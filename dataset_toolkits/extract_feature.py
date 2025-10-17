import argparse
import copy
import importlib
import json
import os
import sys

from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import utils3d

from easydict import EasyDict as edict
from p_tqdm import p_umap
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


torch.set_grad_enabled(False)


def get_data(frames, sha256):
    with ThreadPoolExecutor(max_workers=16) as executor:

        def worker(view):
            image_path = os.path.join(opt.output_dir, "renders", sha256, view["file_path"])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view["transform_matrix"])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view["camera_angle_x"]
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {"image": image, "extrinsics": extrinsics, "intrinsics": intrinsics}

        datas = executor.map(worker, frames)
    for data in datas:
        if data is not None:
            yield data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg", help="Feature extraction model")
    parser.add_argument("--instances", type=str, default=None, help="Instances to process")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=8)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    # Pin this process to a single GPU so compute stays on the intended device
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_idx)

    feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, "features", feature_name), exist_ok=True)

    # load model
    dinov2_model = torch.hub.load("facebookresearch/dinov2", opt.model, verbose=False)
    dinov2_model.eval().cuda()
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    n_patch = 518 // 14

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))
    else:
        raise ValueError("metadata.csv not found")

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    records = []

    if opt.instances is not None:
        with open(opt.instances, "r") as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata["sha256"].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score]
        if f"feature_{feature_name}" in metadata.columns:
            metadata = metadata[metadata[f"feature_{feature_name}"] == False]
        metadata = metadata[metadata["voxelized"] == True]
        metadata = metadata[metadata["rendered"] == True]

    # filter out objects that are already processed
    sha256s = list(metadata["sha256"].values)
    for sha256 in copy.copy(sha256s):
        features_path = os.path.join(opt.output_dir, "features", feature_name, f"{sha256}.npz")
        if os.path.exists(features_path):
            records.append({"sha256": sha256, f"feature_{feature_name}": True})
            sha256s.remove(sha256)

    # Further shard by GPU index for data-parallel launch
    sha256s = sha256s[(opt.gpu_idx % opt.gpu_num) :: opt.gpu_num]

    # extract features sequentially per GPU index
    for sha256 in tqdm(
        sha256s,
        desc=f"GPU {opt.gpu_idx} - Extracting features",
        position=int(opt.gpu_idx),
        leave=True,
    ):
        try:
            with open(os.path.join(opt.output_dir, "renders", sha256, "transforms.json"), "r") as f:
                meta_json = json.load(f)
            frames = meta_json["frames"]
            data = []
            for datum in get_data(frames, sha256):
                datum["image"] = transform(datum["image"])
                data.append(datum)
            positions = utils3d.io.read_ply(os.path.join(opt.output_dir, "voxels", f"{sha256}.ply"))[0]

            positions = torch.from_numpy(positions).float().cuda()
            indices = ((positions + 0.5) * 64).long()
            assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
            n_views = len(data)
            N = positions.shape[0]
            pack = {
                "indices": indices.cpu().numpy().astype(np.uint8),
            }
            patchtokens_lst = []
            uv_lst = []
            for i in range(0, n_views, opt.batch_size):
                batch_data = data[i : i + opt.batch_size]
                bs = len(batch_data)
                batch_images = torch.stack([d["image"] for d in batch_data]).cuda()
                batch_extrinsics = torch.stack([d["extrinsics"] for d in batch_data]).cuda()
                batch_intrinsics = torch.stack([d["intrinsics"] for d in batch_data]).cuda()
                features = dinov2_model(batch_images, is_training=True)
                uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                patchtokens = (
                    features["x_prenorm"][:, dinov2_model.num_register_tokens + 1 :]
                    .permute(0, 2, 1)
                    .reshape(bs, 1024, n_patch, n_patch)
                )
                patchtokens_lst.append(patchtokens)
                uv_lst.append(uv)
            patchtokens = torch.cat(patchtokens_lst, dim=0)
            uv = torch.cat(uv_lst, dim=0)

            # sample patchtokens at projected uv and save
            sampled = (
                F.grid_sample(
                    patchtokens,
                    uv.unsqueeze(1),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(2)
                .permute(0, 2, 1)
                .cpu()
                .numpy()
            )
            pack["patchtokens"] = np.mean(sampled, axis=0).astype(np.float16)
            save_path = os.path.join(opt.output_dir, "features", feature_name, f"{sha256}.npz")
            np.savez_compressed(save_path, **pack)
            records.append({"sha256": sha256, f"feature_{feature_name}": True})
        except Exception as e:
            print(f"Error processing {sha256}: {e}")

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f"feature_{feature_name}_{opt.rank}_{opt.gpu_idx}.csv"), index=False)
