import argparse
import os

from concurrent.futures import ThreadPoolExecutor

import objaverse.xl as oxl
import pandas as pd

from objaverse.xl.github import shutil
from p_tqdm import p_umap
from tqdm import tqdm, trange
from utils import get_file_hash


FLORA_4D_DATA_ROOT = "/viscam/data/4DStateMachine_synthetic_data/flora125_rendered"


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--split", type=str, default="train", help="Data split")


def get_metadata(split, **kwargs):
    # meta_file_name = f"growth_4d_data_flora_{split}.csv"
    meta_file_name = f"flora125_{split}_data.csv"
    metadata_path = os.path.join(FLORA_4D_DATA_ROOT, meta_file_name)
    assert os.path.exists(metadata_path), f"Metadata file {metadata_path} does not exist"

    print(f"Loading metadata from {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    return metadata


def download(metadata, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)

    file_paths = {}
    for i in trange(len(metadata), desc="Downloading objects"):
        src_path = os.path.join(FLORA_4D_DATA_ROOT, metadata.iloc[i]["raw_file_identifier"])
        dst_path = os.path.join(output_dir, "raw", metadata.iloc[i]["sha256"] + ".obj")
        shutil.copy(src_path, dst_path)
        file_paths[metadata.iloc[i]["sha256"]] = os.path.relpath(dst_path, output_dir)

    downloaded = {}
    for k, v in file_paths.items():
        downloaded[k] = v

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc="Processing objects") -> pd.DataFrame:
    # load metadata
    metadata = metadata.to_dict("records")

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    # try:
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(metadata), desc=desc) as pbar:

    #         def worker(metadatum):
    #             try:
    #                 local_path = metadatum["local_path"]
    #                 sha256 = metadatum["sha256"]
    #                 file = os.path.join(output_dir, local_path)
    #                 record = func(file, sha256)
    #                 if record is not None:
    #                     records.append(record)
    #                 pbar.update()
    #             except Exception as e:
    #                 print(f"Error processing object {sha256}: {e}")
    #                 pbar.update()

    #         executor.map(worker, metadata)
    #         executor.shutdown(wait=True)
    # except:
    #     print("Error happened during processing.")

    def worker(metadatum):
        try:
            local_path = metadatum["local_path"]
            sha256 = metadatum["sha256"]
            file = os.path.join(output_dir, local_path)
            record = func(file, sha256)
            if record is not None:
                records.append(record)

        except Exception as e:
            print(f"Error processing object {sha256}: {e}")

    p_umap(worker, metadata, desc=desc, num_cpus=max_workers)

    return pd.DataFrame.from_records(records)
