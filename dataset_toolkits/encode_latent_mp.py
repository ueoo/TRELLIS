import os
import shlex

from argparse import ArgumentParser
from multiprocessing import Pool

import torch


def _run(cmd):
    print(cmd, flush=True)
    os.system(cmd)


def launch_encode_latent_jobs(args):
    # Detect available GPUs; cap by requested gpu_num
    available_gpus = max(1, torch.cuda.device_count())

    # Build per-GPU commands to invoke encode_latent.py
    cmds = []
    if args.gpu_ids is not None:
        gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
        gpu_num = len(gpu_ids)
    else:
        gpu_ids = list(range(args.gpu_num))
        gpu_num = args.gpu_num
    for gpu_idx in gpu_ids:
        env = f"CUDA_VISIBLE_DEVICES={gpu_idx%available_gpus}"
        cpu_env = (
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 " "NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1"
        )
        cmd = (
            f"export {env} {cpu_env} && python dataset_toolkits/encode_latent.py "
            f"--output_dir {shlex.quote(args.output_dir)} --feat_model {shlex.quote(args.feat_model)} "
            f"--rank {args.rank} --world_size {args.world_size} "
            f"--gpu_idx {gpu_idx} --gpu_num {gpu_num}"
        )
        if args.instances is not None:
            cmd += f" --instances {shlex.quote(args.instances)}"
        if args.filter_low_aesthetic_score is not None:
            cmd += f" --filter_low_aesthetic_score {args.filter_low_aesthetic_score}"
        if args.enc_pretrained is not None:
            cmd += f" --enc_pretrained {shlex.quote(args.enc_pretrained)}"
        if args.enc_model is not None:
            cmd += f" --enc_model {shlex.quote(args.enc_model)}"
        if args.ckpt is not None:
            cmd += f" --ckpt {shlex.quote(args.ckpt)}"
        if args.model_root is not None:
            cmd += f" --model_root {shlex.quote(args.model_root)}"
        cmds.append(cmd)

    with Pool(gpu_num) as pool:
        pool.map(_run, cmds)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feat_model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument(
        "--enc_pretrained", type=str, default="microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
    )
    parser.add_argument("--enc_model", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--model_root", type=str, default="results")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--filter_low_aesthetic_score", type=float, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--gpu_ids", type=str, default=None)
    args = parser.parse_args()

    launch_encode_latent_jobs(args)
