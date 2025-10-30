import os

from argparse import ArgumentParser
from multiprocessing import Pool

import torch


def _run(cmd):
    print(cmd, flush=True)
    os.system(cmd)


def launch_render_jobs(args):
    # Detect available GPUs; cap by requested gpu_num
    available_gpus = max(1, torch.cuda.device_count())

    # Build per-GPU commands to invoke render_fixview.py
    cmds = []
    for gpu_idx in range(available_gpus):
        env = f"CUDA_VISIBLE_DEVICES={gpu_idx}"
        cpu_env = (
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 " "NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1"
        )
        cmd = (
            f"export {env} {cpu_env} && python"
            f" inference/flora_4d_geo_bench_gtprev_test_scenes.py"
            f" --rank {args.rank} --world_size {args.world_size}"
            f" --gpu_idx {gpu_idx} --gpu_num {available_gpus}"
        )
        cmds.append(cmd)

    with Pool(available_gpus) as pool:
        pool.map(_run, cmds)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    launch_render_jobs(args)
