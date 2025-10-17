import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from easydict import EasyDict as edict
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import trellis.models as models
import trellis.modules.sparse as sp


torch.set_grad_enabled(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument("--feat_model", type=str, default="dinov2_vitl14_reg", help="Feature model")
    parser.add_argument(
        "--enc_pretrained",
        type=str,
        default="microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16",
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
        latent_name = f'{opt.feat_model}_{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f"{opt.feat_model}_{opt.enc_model}_{opt.ckpt}"
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, "config.json"), "r")))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, "ckpts", f"encoder_{opt.ckpt}.pt")
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f"Loaded model from {ckpt_path}")

    os.makedirs(os.path.join(opt.output_dir, "latents", latent_name), exist_ok=True)

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
            sha256s = [line.strip() for line in f]
        metadata = metadata[metadata["sha256"].isin(sha256s)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata[f"feature_{opt.feat_model}"] == True]
        if f"latent_{latent_name}" in metadata.columns:
            metadata = metadata[metadata[f"latent_{latent_name}"] == False]

    # filter out objects that are already processed
    sha256s = list(metadata["sha256"].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, "latents", latent_name, f"{sha256}.npz")):
            records.append({"sha256": sha256, f"latent_{latent_name}": True})
            sha256s.remove(sha256)

    # Further shard by GPU index for data-parallel launch
    sha256s = sha256s[(opt.gpu_idx % opt.gpu_num) :: opt.gpu_num]
    # encode latents sequentially per GPU index
    for sha256 in tqdm(
        sha256s,
        desc=f"GPU {opt.gpu_idx} - Extracting latents",
        position=int(opt.gpu_idx),
        leave=True,
    ):
        try:
            feats_np = np.load(os.path.join(opt.output_dir, "features", opt.feat_model, f"{sha256}.npz"))
            sparse_feats = sp.SparseTensor(
                feats=torch.from_numpy(feats_np["patchtokens"]).float(),
                coords=torch.cat(
                    [
                        torch.zeros(feats_np["patchtokens"].shape[0], 1).int(),
                        torch.from_numpy(feats_np["indices"]).int(),
                    ],
                    dim=1,
                ),
            ).cuda()
            latent = encoder(sparse_feats, sample_posterior=False)
            assert torch.isfinite(latent.feats).all(), "Non-finite latent"
            pack = {
                "feats": latent.feats.cpu().numpy().astype(np.float32),
                "coords": latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
            }
            save_path = os.path.join(opt.output_dir, "latents", latent_name, f"{sha256}.npz")
            np.savez_compressed(save_path, **pack)
            records.append({"sha256": sha256, f"latent_{latent_name}": True})
        except Exception as e:
            print(f"Error processing {sha256}: {e}")

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f"latent_{latent_name}_{opt.rank}_{opt.gpu_idx}.csv"), index=False)
