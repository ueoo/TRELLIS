import argparse
import copy
import importlib
import os
import sys

from functools import partial

import numpy as np
import open3d as o3d
import pandas as pd
import utils3d

from easydict import EasyDict as edict


def _voxelize(file, sha256, output_dir):
    mesh_path = os.path.join(output_dir, "renders", sha256, "mesh.ply")
    if not os.path.exists(mesh_path):
        print(f"Mesh file not found for {sha256}")
        return {"sha256": sha256, "voxelized": False, "num_voxels": 0}
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    except Exception as e:
        print(f"Error reading mesh file for {sha256}: {e}")
        return {"sha256": sha256, "voxelized": False, "num_voxels": 0}
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1 / 64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5)
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    if np.any(vertices < 0) or np.any(vertices >= 64):
        print(f"Some vertices are out of bounds for {sha256}")
        return {"sha256": sha256, "voxelized": False, "num_voxels": 0}
    vertices = (vertices + 0.5) / 64 - 0.5
    out_path = os.path.join(output_dir, "voxels", f"{sha256}.ply")
    utils3d.io.write_ply(out_path, vertices)
    if not os.path.exists(out_path):
        print(f"Failed to write voxelized mesh for {sha256}")
        return {"sha256": sha256, "voxelized": False, "num_voxels": 0}
    return {"sha256": sha256, "voxelized": True, "num_voxels": len(vertices)}


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
    dataset_utils.add_args(parser)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=None)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, "voxels"), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score]
        if "rendered" not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata["rendered"] == True]
        if "voxelized" in metadata.columns:
            metadata = metadata[metadata["voxelized"] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata["sha256"].values):
        voxel_file_path = os.path.join(opt.output_dir, "voxels", f"{sha256}.ply")
        if os.path.exists(voxel_file_path):
            pts = utils3d.io.read_ply(voxel_file_path)[0]
            records.append({"sha256": sha256, "voxelized": True, "num_voxels": len(pts)})
            metadata = metadata[metadata["sha256"] != sha256]

    print(f"Voxelizing {len(metadata)} objects...")

    # process objects
    func = partial(_voxelize, output_dir=opt.output_dir)
    voxelized = dataset_utils.foreach_instance(
        metadata, opt.output_dir, func, max_workers=opt.max_workers, desc="Voxelizing"
    )
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(opt.output_dir, f"voxelized_{opt.rank}.csv"), index=False)
