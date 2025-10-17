import argparse
import importlib
import os
import shutil
import sys
import time

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import utils3d

from easydict import EasyDict as edict
from p_tqdm import p_umap
from tqdm import tqdm


def get_first_directory(path):
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                return entry.name
    return None


def need_process(key):
    return key in opt.field or opt.field == ["all"]


def merge_csv_by_prefix(metadata, output_dir, filename_prefix, timestamp, downloaded_mode=False):
    df_files = [f for f in os.listdir(output_dir) if f.startswith(filename_prefix) and f.endswith(".csv")]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        # Concatenate and ensure unique rows per sha256
        df = pd.concat(df_parts, ignore_index=True)
        if "sha256" in df.columns:
            df = df.drop_duplicates(subset=["sha256"], keep="last")
            df.set_index("sha256", inplace=True)
        else:
            # If sha256 is already the index, ensure uniqueness
            if df.index.name == "sha256":
                df = df[~df.index.duplicated(keep="last")]
        # Final guard against duplicate index labels
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        if downloaded_mode:
            if "local_path" in metadata.columns:
                metadata.update(df, overwrite=True)
            else:
                # Join on index (both are indexed by sha256)
                metadata = metadata.join(df, how="left")
        else:
            metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(output_dir, f), os.path.join(output_dir, "merged_records", f"{timestamp}_{f}"))
    return metadata


if __name__ == "__main__":
    dataset_utils = importlib.import_module(f"datasets.{sys.argv[1]}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument("--field", type=str, default="all", help="Fields to process, separated by commas")
    parser.add_argument(
        "--from_file",
        action="store_true",
        help="Build metadata from file instead of from records of processings."
        + "Useful when some processing fail to generate records but file already exists.",
    )
    dataset_utils.add_args(parser)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "merged_records"), exist_ok=True)

    opt.field = opt.field.split(",")

    timestamp = str(int(time.time()))

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        print("Loading previous metadata...")
        metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"), low_memory=False)
    else:
        metadata = dataset_utils.get_metadata(**opt)
    # Ensure no duplicate sha256 before indexing to avoid duplicate-label issues later
    if "sha256" in metadata.columns:
        metadata = metadata.drop_duplicates(subset=["sha256"], keep="last")
        metadata.set_index("sha256", inplace=True)
    else:
        if metadata.index.name == "sha256":
            metadata = metadata[~metadata.index.duplicated(keep="last")]

    # merge downloaded
    metadata = merge_csv_by_prefix(metadata, opt.output_dir, "downloaded_", timestamp, downloaded_mode=True)

    # detect models
    image_models = []
    if os.path.exists(os.path.join(opt.output_dir, "features")):
        image_models = os.listdir(os.path.join(opt.output_dir, "features"))
    latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, "latents")):
        latent_models = os.listdir(os.path.join(opt.output_dir, "latents"))
    ss_latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, "ss_latents")):
        ss_latent_models = os.listdir(os.path.join(opt.output_dir, "ss_latents"))
    print(f"Image models: {image_models}")
    print(f"Latent models: {latent_models}")
    print(f"Sparse Structure latent models: {ss_latent_models}")

    if "rendered" not in metadata.columns:
        metadata["rendered"] = [False] * len(metadata)
    if "voxelized" not in metadata.columns:
        metadata["voxelized"] = [False] * len(metadata)
    if "num_voxels" not in metadata.columns:
        metadata["num_voxels"] = [0] * len(metadata)
    if "fixview_rendered" not in metadata.columns:
        metadata["fixview_rendered"] = [False] * len(metadata)
    if "cond_rendered" not in metadata.columns:
        metadata["cond_rendered"] = [False] * len(metadata)
    if "cond_rendered_test" not in metadata.columns:
        metadata["cond_rendered_test"] = [False] * len(metadata)
    for model in image_models:
        if f"feature_{model}" not in metadata.columns:
            metadata[f"feature_{model}"] = [False] * len(metadata)
    for model in latent_models:
        if f"latent_{model}" not in metadata.columns:
            metadata[f"latent_{model}"] = [False] * len(metadata)
    for model in ss_latent_models:
        if f"ss_latent_{model}" not in metadata.columns:
            metadata[f"ss_latent_{model}"] = [False] * len(metadata)

    # merge rendered
    metadata = merge_csv_by_prefix(metadata, opt.output_dir, "rendered_", timestamp)

    # merge voxelized
    metadata = merge_csv_by_prefix(metadata, opt.output_dir, "voxelized_", timestamp)

    # merge fixview_rendered
    metadata = merge_csv_by_prefix(metadata, opt.output_dir, "fixview_rendered_", timestamp)

    # merge cond_rendered
    metadata = merge_csv_by_prefix(metadata, opt.output_dir, "cond_rendered_", timestamp)

    # merge cond_rendered
    metadata = merge_csv_by_prefix(metadata, opt.output_dir, "cond_rendered_test_", timestamp)

    # merge features
    for model in image_models:
        metadata = merge_csv_by_prefix(metadata, opt.output_dir, f"feature_{model}_", timestamp)

    # merge latents
    for model in latent_models:
        metadata = merge_csv_by_prefix(metadata, opt.output_dir, f"latent_{model}_", timestamp)

    # merge sparse structure latents
    for model in ss_latent_models:
        metadata = merge_csv_by_prefix(metadata, opt.output_dir, f"ss_latent_{model}_", timestamp)

    # build metadata from files
    if opt.from_file:
        def worker(sha256):
            if (
                need_process("rendered")
                and metadata.loc[sha256, "rendered"] == False
                and os.path.exists(os.path.join(opt.output_dir, "renders", sha256, "transforms.json"))
                and os.path.exists(os.path.join(opt.output_dir, "renders", sha256, "mesh.ply"))
            ):
                metadata.loc[sha256, "rendered"] = True
            if (
                need_process("voxelized")
                and metadata.loc[sha256, "rendered"] == True
                and metadata.loc[sha256, "voxelized"] == False
                and os.path.exists(os.path.join(opt.output_dir, "voxels", f"{sha256}.ply"))
            ):
                try:
                    pts = utils3d.io.read_ply(os.path.join(opt.output_dir, "voxels", f"{sha256}.ply"))[0]
                    metadata.loc[sha256, "voxelized"] = True
                    metadata.loc[sha256, "num_voxels"] = len(pts)
                except Exception as e:
                    pass

            if (
                need_process("fixview_rendered")
                and metadata.loc[sha256, "fixview_rendered"] == False
                and os.path.exists(os.path.join(opt.output_dir, "renders_fixview", sha256, "transforms.json"))
            ):
                metadata.loc[sha256, "fixview_rendered"] = True

            if (
                need_process("cond_rendered")
                and metadata.loc[sha256, "cond_rendered"] == False
                and os.path.exists(os.path.join(opt.output_dir, "renders_cond", sha256, "transforms.json"))
            ):
                metadata.loc[sha256, "cond_rendered"] = True

            if (
                need_process("cond_rendered_test")
                and metadata.loc[sha256, "cond_rendered_test"] == False
                and os.path.exists(os.path.join(opt.output_dir, "renders_cond_test", sha256, "transforms.json"))
            ):
                metadata.loc[sha256, "cond_rendered_test"] = True

            for model in image_models:
                if (
                    need_process(f"feature_{model}")
                    and metadata.loc[sha256, f"feature_{model}"] == False
                    and metadata.loc[sha256, "rendered"] == True
                    and metadata.loc[sha256, "voxelized"] == True
                    and os.path.exists(os.path.join(opt.output_dir, "features", model, f"{sha256}.npz"))
                ):
                    metadata.loc[sha256, f"feature_{model}"] = True
            for model in latent_models:
                if (
                    need_process(f"latent_{model}")
                    and metadata.loc[sha256, f"latent_{model}"] == False
                    and metadata.loc[sha256, "rendered"] == True
                    and metadata.loc[sha256, "voxelized"] == True
                    and os.path.exists(os.path.join(opt.output_dir, "latents", model, f"{sha256}.npz"))
                ):
                    metadata.loc[sha256, f"latent_{model}"] = True
            for model in ss_latent_models:
                if (
                    need_process(f"ss_latent_{model}")
                    and metadata.loc[sha256, f"ss_latent_{model}"] == False
                    and metadata.loc[sha256, "voxelized"] == True
                    and os.path.exists(os.path.join(opt.output_dir, "ss_latents", model, f"{sha256}.npz"))
                ):
                    metadata.loc[sha256, f"ss_latent_{model}"] = True

        p_umap(worker, metadata.index, desc="Building metadata", num_cpus=os.cpu_count())

        #         pbar.update()
        #     except Exception as e:
        #         print(f"Error processing {sha256}: {e}")
        #         pbar.update()

        # executor.map(worker, metadata.index)
        # executor.shutdown(wait=True)

    # statistics
    metadata.to_csv(os.path.join(opt.output_dir, "metadata.csv"))
    num_downloaded = metadata["local_path"].count() if "local_path" in metadata.columns else 0
    with open(os.path.join(opt.output_dir, "statistics.txt"), "w") as f:
        f.write("Statistics:\n")
        f.write(f"  - Number of assets: {len(metadata)}\n")
        f.write(f"  - Number of assets downloaded: {num_downloaded}\n")
        f.write(f'  - Number of assets rendered: {metadata["rendered"].sum()}\n')
        f.write(f'  - Number of assets voxelized: {metadata["voxelized"].sum()}\n')
        if len(image_models) != 0:
            f.write(f"  - Number of assets with image features extracted:\n")
            for model in image_models:
                f.write(f'    - {model}: {metadata[f"feature_{model}"].sum()}\n')
        if len(latent_models) != 0:
            f.write(f"  - Number of assets with latents extracted:\n")
            for model in latent_models:
                f.write(f'    - {model}: {metadata[f"latent_{model}"].sum()}\n')
        if len(ss_latent_models) != 0:
            f.write(f"  - Number of assets with sparse structure latents extracted:\n")
            for model in ss_latent_models:
                f.write(f'    - {model}: {metadata[f"ss_latent_{model}"].sum()}\n')
        f.write(f'  - Number of assets with captions: {metadata["captions"].count()}\n')
        f.write(f'  - Number of assets with fixview rendered: {metadata["fixview_rendered"].sum()}\n')
        f.write(f'  - Number of assets with image conditions: {metadata["cond_rendered"].sum()}\n')
        f.write(f'  - Number of assets with test image conditions: {metadata["cond_rendered_test"].sum()}\n')

    with open(os.path.join(opt.output_dir, "statistics.txt"), "r") as f:
        print(f.read())
