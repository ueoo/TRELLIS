import argparse
import copy
import importlib
import json
import os
import sys

from functools import partial
from subprocess import DEVNULL, call

import numpy as np
import pandas as pd

from easydict import EasyDict as edict
from tqdm import tqdm
from utils import sphere_hammersley_sequence


BLENDER_LINK = "https://download.blender.org/release/Blender4.5/blender-4.5.2-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = "/svl/u/yuegao/blender"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-4.5.2-linux-x64/blender"


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        # os.system("sudo apt-get update")
        # os.system("sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6")
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(
            f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.2-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}"
        )


def _render_fixview(file_path, sha256, output_dir, r=1.5, num_views=10, selected_views=[3, 6, 8]):
    if isinstance(selected_views, str):
        selected_views = [int(i) for i in selected_views.split(",")]
    assert len(selected_views) > 0 and len(selected_views) <= num_views
    assert all(i >= 0 and i < num_views for i in selected_views)
    output_folder = os.path.join(output_dir, "renders_fixview", sha256)

    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (0.0, 0.0)
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [r] * num_views
    fov = [40 / 180 * np.pi] * num_views

    views = [{"yaw": y, "pitch": p, "radius": r, "fov": f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    views = [views[i] for i in selected_views]

    args = [
        BLENDER_PATH,
        "-b",
        "-P",
        os.path.join(os.path.dirname(__file__), "blender_script", "render.py"),
        "--",
        "--views",
        json.dumps(views),
        "--object",
        os.path.expanduser(file_path),
        "--output_folder",
        output_folder,
        "--engine",
        "CYCLES",
        "--resolution",
        "1024",
    ]
    if file_path.endswith(".blend"):
        args.insert(1, file_path)

    call(args, stdout=DEVNULL, stderr=DEVNULL)
    # call(args)

    json_path = os.path.join(output_folder, "transforms.json")
    if os.path.exists(json_path):
        return {"sha256": sha256, "fixview_rendered": True}


if __name__ == "__main__":
    dataset_utils = importlib.import_module(f"datasets.{sys.argv[1]}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument("--instances", type=str, default=None, help="Instances to process")
    parser.add_argument("--num_views", type=int, default=10, help="Number of views to render")
    parser.add_argument("--selected_views", type=str, default="3,6,8", help="Selected views to render")
    parser.add_argument("--radius", type=float, default=1.5, help="Radius of the camera")
    dataset_utils.add_args(parser)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Pin this process to a single GPU (so Blender uses the intended device)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_idx)

    os.makedirs(os.path.join(opt.output_dir, "renders_fixview"), exist_ok=True)

    # install blender
    print("Checking blender...", flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    records = []

    if opt.instances is None:
        metadata = metadata[metadata["local_path"].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score]
        # if "fixview_rendered" in metadata.columns:
        #     metadata = metadata[metadata["fixview_rendered"] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    actual_num_views = opt.selected_views.count(",") + 1
    # filter out objects that are already processed
    for sha256 in copy.copy(metadata["sha256"].values):
        json_path = os.path.join(opt.output_dir, "renders_fixview", sha256, "transforms.json")
        render_folder = os.path.join(opt.output_dir, "renders_fixview", sha256)
        rendered_frames = [f for f in os.listdir(render_folder) if f.endswith(".png")]
        if len(rendered_frames) == actual_num_views and os.path.exists(json_path):
            records.append({"sha256": sha256, "fixview_rendered": True})
            metadata = metadata[metadata["sha256"] != sha256]
        else:
            print(f"Object {sha256} not fully rendered")

    print(f"Rendering {len(metadata)} objects with fixed views...")

    metadata = metadata[opt.gpu_idx :: opt.gpu_num]

    # Process objects with a simple for-loop on the assigned GPU
    metadata = metadata.to_dict("records")
    for metadatum in tqdm(
        metadata,
        desc=f"GPU {opt.gpu_idx}",
        position=int(opt.gpu_idx),
        leave=True,
    ):
        file_path = os.path.join(opt.output_dir, metadatum["local_path"])
        record = _render_fixview(
            file_path,
            metadatum["sha256"],
            opt.output_dir,
            opt.radius,
            opt.num_views,
            opt.selected_views,
        )
        if record is not None:
            records.append(record)

    rendered_df = pd.DataFrame.from_records(records)
    rendered_df.to_csv(os.path.join(opt.output_dir, f"fixview_rendered_{opt.rank}_{opt.gpu_idx}.csv"), index=False)
