import argparse
import os

from concurrent.futures import ThreadPoolExecutor

import objaverse.xl as oxl
import pandas as pd

from objaverse.xl.github import shutil
from tqdm import tqdm, trange
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(split="train", **kwargs):
    if split == "train":
        metadata_path = "/svl/u/yuegao/4DStateMachine/growth_4d_scenes/growth_4d_data_train.csv"
    elif split == "test":
        metadata_path = "/svl/u/yuegao/4DStateMachine/growth_4d_scenes/growth_4d_data_test.csv"
    elif split == "debug":
        metadata_path = "/svl/u/yuegao/4DStateMachine/growth_4d_scenes/growth_4d_data_debug.csv"
    else:
        metadata_path = "/svl/u/yuegao/4DStateMachine/growth_4d_scenes/growth_4d_data.csv"

    print(f"Loading metadata from {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    return metadata


def download(metadata, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)

    original_path = "/svl/u/yuegao/4DStateMachine/growth_4d_scenes"

    file_paths = {}
    for i in trange(len(metadata), desc="Downloading objects"):
        src_path = os.path.join(original_path, metadata.iloc[i]["raw_file_identifier"])
        dst_path = os.path.join(output_dir, "raw", metadata.iloc[i]["sha256"] + ".obj")
        shutil.copy(src_path, dst_path)
        file_paths[metadata.iloc[i]["sha256"]] = os.path.relpath(dst_path, output_dir)

    downloaded = {}
    for k, v in file_paths.items():
        downloaded[k] = v

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc="Processing objects") -> pd.DataFrame:
    import os
    import tempfile
    import zipfile

    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    # load metadata
    metadata = metadata.to_dict("records")

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(metadata), desc=desc) as pbar:

            def worker(metadatum):
                try:
                    local_path = metadatum["local_path"]
                    sha256 = metadatum["sha256"]
                    file = os.path.join(output_dir, local_path)
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()

            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")

    return pd.DataFrame.from_records(records)
