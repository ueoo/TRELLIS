import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import utils3d

from easydict import EasyDict as edict
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import trellis.models as models


torch.set_grad_enabled(False)


def get_voxels(instance):
    position = utils3d.io.read_ply(os.path.join(opt.output_dir, "voxels", f"{instance}.ply"))[0]
    coords = ((torch.tensor(position) + 0.5) * opt.resolution).int().contiguous()
    ss = torch.zeros(1, opt.resolution, opt.resolution, opt.resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument(
        "--enc_pretrained",
        type=str,
        default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
        help="Pretrained encoder model",
    )
    parser.add_argument("--model_root", type=str, default="results", help="Root directory of models")
    parser.add_argument(
        "--enc_model",
        type=str,
        default=None,
        help="Encoder model. if specified, use this model instead of pretrained model",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--resolution", type=int, default=64, help="Resolution")
    parser.add_argument("--instances", type=str, default=None, help="Instances to process")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=8)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    # Pin this process to a single GPU so compute stays on the intended device
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_idx)

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f"{opt.enc_model}_{opt.ckpt}"
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, "config.json"), "r")))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, "ckpts", f"encoder_{opt.ckpt}.pt")
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f"Loaded model from {ckpt_path}")

    os.makedirs(os.path.join(opt.output_dir, "ss_latents", latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))
    else:
        raise ValueError("metadata.csv not found")

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    # Further shard by GPU index for data-parallel launch
    metadata = metadata[opt.gpu_idx :: opt.gpu_num]
    records = []

    if opt.instances is not None:
        with open(opt.instances, "r") as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata["sha256"].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata["voxelized"] == True]
        if f"ss_latent_{latent_name}" in metadata.columns:
            metadata = metadata[metadata[f"ss_latent_{latent_name}"] == False]

    # filter out objects that are already processed
    sha256s = list(metadata["sha256"].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, "ss_latents", latent_name, f"{sha256}.npz")):
            records.append({"sha256": sha256, f"ss_latent_{latent_name}": True})
            sha256s.remove(sha256)

    # encode latents sequentially per GPU index
    for sha256 in tqdm(sha256s, desc=f"GPU {opt.gpu_idx} - Extracting ss latents"):
        try:
            ss = get_voxels(sha256)[None].float()
            ss = ss.cuda().float()
            latent = encoder(ss, sample_posterior=False)
            assert torch.isfinite(latent).all(), "Non-finite latent"
            pack = {
                "mean": latent[0].cpu().numpy(),
            }
            save_path = os.path.join(opt.output_dir, "ss_latents", latent_name, f"{sha256}.npz")
            np.savez_compressed(save_path, **pack)
            records.append({"sha256": sha256, f"ss_latent_{latent_name}": True})
        except Exception as e:
            print(f"Error processing {sha256}: {e}")

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f"ss_latent_{latent_name}_{opt.rank}_{opt.gpu_idx}.csv"), index=False)
