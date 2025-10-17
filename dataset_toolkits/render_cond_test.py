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
from utils import sphere_hammersley_sequence


BLENDER_LINK = "https://download.blender.org/release/Blender4.5/blender-4.5.2-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-4.5.2-linux-x64/blender"


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        # os.system("sudo apt-get update")
        # os.system("sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6")
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(
            f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.2-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}"
        )


def _render_cond_test(file_path, sha256, output_dir, num_views):
    output_folder = os.path.join(output_dir, "renders_cond_test", sha256)

    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    # fov_min, fov_max = 10, 70
    # radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    # radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    # k_min = 1 / radius_max**2
    # k_max = 1 / radius_min**2
    # ks = np.random.uniform(k_min, k_max, (1000000,))
    # radius = [1 / np.sqrt(k) for k in ks]
    # fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]

    # for regrowth, we simplify the fov and radius
    fov_min, fov_max = 30, 50
    radius_min, radius_max = 0.5, 2.0
    radius = [np.random.uniform(radius_min, radius_max) for _ in range(num_views)]
    fov = [np.random.uniform(fov_min, fov_max) for _ in range(num_views)]
    fov = [f / 180 * np.pi for f in fov]
    views = [{"yaw": y, "pitch": p, "radius": r, "fov": f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    args = [
        BLENDER_PATH,
        "-b",
        "-P",
        os.path.join(os.path.dirname(__file__), "blender_script", "render.py"),
        "--",
        "--views",
        json.dumps(views, default=lambda o: float(o) if hasattr(o, "item") else o),
        "--object",
        os.path.expanduser(file_path),
        "--output_folder",
        os.path.expanduser(output_folder),
        "--resolution",
        "1024",
    ]
    if file_path.endswith(".blend"):
        args.insert(1, file_path)

    call(args, stdout=DEVNULL)

    if os.path.exists(os.path.join(output_folder, "transforms.json")):
        return {"sha256": sha256, "cond_rendered_test": True}


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
    parser.add_argument("--num_views", type=int, default=24, help="Number of views to render")
    dataset_utils.add_args(parser)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, "renders_cond_test"), exist_ok=True)

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
        if "cond_rendered_test" in metadata.columns:
            metadata = metadata[metadata["cond_rendered_test"] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata["sha256"].values):
        if os.path.exists(os.path.join(opt.output_dir, "renders_cond_test", sha256, "transforms.json")):
            records.append({"sha256": sha256, "cond_rendered_test": True})
            metadata = metadata[metadata["sha256"] != sha256]

    print(f"Processing {len(metadata)} objects...")

    # process objects
    func = partial(_render_cond_test, output_dir=opt.output_dir, num_views=opt.num_views)
    cond_rendered = dataset_utils.foreach_instance(
        metadata, opt.output_dir, func, max_workers=opt.max_workers, desc="Rendering objects"
    )
    cond_rendered = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
    cond_rendered.to_csv(os.path.join(opt.output_dir, f"test_cond_rendered_{opt.rank}.csv"), index=False)
