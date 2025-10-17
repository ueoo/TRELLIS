import os
import shlex
import socket

from argparse import ArgumentParser
from multiprocessing import Pool

import torch


def _run(cmd):
    print(cmd, flush=True)
    os.system(cmd)


def launch_feature_jobs(args):
    # Detect available GPUs; cap by requested gpu_num
    # available_gpus = max(1, torch.cuda.device_count())

    # Build per-GPU commands to invoke extract_feature.py
    cmds = []
    if args.gpu_ids is not None:
        gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
        gpu_num = len(gpu_ids)
    else:
        gpu_ids = list(range(args.gpu_num))
        gpu_num = args.gpu_num
    for gpu_idx in gpu_ids:
        env = f"CUDA_VISIBLE_DEVICES={gpu_idx}"
        cpu_env = (
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 " "NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1"
        )
        cmd = (
            f"export {env} {cpu_env} && python dataset_toolkits/extract_feature.py "
            f"--output_dir {shlex.quote(args.output_dir)} --model {shlex.quote(args.model)} "
            f"--rank {args.rank} --world_size {args.world_size} --batch_size {args.batch_size} "
            f"--gpu_idx {gpu_idx} --gpu_num {gpu_num}"
        )
        if args.instances is not None:
            cmd += f" --instances {shlex.quote(args.instances)}"
        if args.filter_low_aesthetic_score is not None:
            cmd += f" --filter_low_aesthetic_score {args.filter_low_aesthetic_score}"
        cmds.append(cmd)

    with Pool(gpu_num) as pool:
        pool.map(_run, cmds)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--filter_low_aesthetic_score", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--gpu_ids", type=str, default=None)
    args = parser.parse_args()

    launch_feature_jobs(args)
