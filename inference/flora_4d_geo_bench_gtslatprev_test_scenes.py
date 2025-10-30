import argparse
import os
import shutil


# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import json
import sys

import imageio
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from trellis.pipelines import TrellisImageTo4DPipeline
from trellis.utils import postprocessing_utils, render_utils


def main(args):

    # Load a pipeline from a model folder or a Hugging Face model hub.
    project_root = "/viscam/projects/4d-state-machine"
    results_root = "/viscam/projects/4d-state-machine/TRELLIS_results"

    finetune_name = "flora4dgeo-pretrainedvae-fullflow"

    pipeline_path = f"{project_root}/TRELLIS-frompretrained/TRELLIS-image-large-{finetune_name}"

    print(f"Loading pipeline from {pipeline_path}")
    pipeline = TrellisImageTo4DPipeline.from_pretrained(pipeline_path)
    pipeline.cuda()

    output_folder = os.path.join(results_root, f"results_gtslatprevcond_{finetune_name.replace('-', '_')}")
    print(f"Saving outputs to {output_folder}")

    infer_data_root = "/scr2/yuegao/TRELLIS_datasets/Flora125Geo_test_merged"

    infer_scene_names = [
        "floraa006",
        # "floraa018",
        # "floraa021",
        "florab010",
        # "florab013",
        # "florab017",
        "florae003",
        # "florae007",
        # "florae017",
        "floraf008",
        # "floraf020",
        # "floraf024",
        "florag004",
        # "florag018",
        # "florag025",
    ]

    # Partition the work similar to dataset_toolkits/render.py
    if args.world_size > 1:
        start = len(infer_scene_names) * args.rank // args.world_size
        end = len(infer_scene_names) * (args.rank + 1) // args.world_size
        infer_scene_names = infer_scene_names[start:end]
    # Further split across GPUs
    infer_scene_names = infer_scene_names[args.gpu_idx :: args.gpu_num]

    num_frames = args.num_frames

    for scene_name in tqdm(
        infer_scene_names,
        total=len(infer_scene_names),
        desc=f"GPU {args.gpu_idx}",
        position=int(args.gpu_idx),
        leave=True,
    ):
        # Load an image
        cond_frame_path = os.path.join(
            infer_data_root, "renders_fixview", f"{scene_name}_{args.first_frame:04d}", "000.png"
        )

        image = Image.open(cond_frame_path)

        ss_latent_folder = os.path.join(infer_data_root, "ss_latents", "ss_enc_conv3d_16l8_fp16")
        ss_latent_prev_folder = os.path.join(infer_data_root, "ss_latents_prev", "ss_enc_conv3d_16l8_fp16")
        assert os.path.exists(ss_latent_folder), f"SS latent path {ss_latent_folder} does not exist"
        assert os.path.exists(ss_latent_prev_folder), f"SS latent path {ss_latent_prev_folder} does not exist"
        slat_latent_folder = os.path.join(infer_data_root, "latents", "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
        assert os.path.exists(slat_latent_folder), f"Slat latent path {slat_latent_folder} does not exist"
        slat_latent_prev_folder = os.path.join(
            infer_data_root, "latents_prev", "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
        )
        assert os.path.exists(slat_latent_prev_folder), f"Slat latent path {slat_latent_prev_folder} does not exist"
        # Run the pipeline
        metrics = pipeline.run_latent_distance(
            image,
            seed=1,
            num_frames=num_frames,
            scene_name=scene_name,
            first_frame=args.first_frame,
            # ss_latent_prev_folder=ss_latent_prev_folder,
            slat_latent_prev_folder=slat_latent_prev_folder,
            ss_latent_folder=ss_latent_folder,
            slat_latent_folder=slat_latent_folder,
        )

        sub_folder = f"sample_gs_{num_frames}_test_scene_{scene_name}_frame_{args.first_frame:04d}_metrics"
        scene_out_dir = os.path.join(output_folder, sub_folder)
        os.makedirs(scene_out_dir, exist_ok=True)

        # # Save metrics as JSON for later analysis
        # with open(os.path.join(scene_out_dir, f"metrics_{scene_name}.json"), "r") as f:
        #     metrics = json.load(f)

        # Extract per-frame metrics
        per_frame = metrics.get("per_frame", [])
        overall = metrics.get("overall", {})
        frames = [pf.get("frame") for pf in per_frame]
        print(f"frames evaluated: {overall.get('frames_evaluated', len(per_frame))}")

        def plot_series(name: str, values: list, ylabel: str):
            if len(values) == 0 or all(v is None for v in values):
                return
            y = [v if v is not None else float("nan") for v in values]
            plt.figure(figsize=(8, 4))
            plt.plot(frames, y, "-o", linewidth=2)
            plt.xlabel("frame")
            plt.ylabel(ylabel)
            plt.title(f"{scene_name} - {ylabel}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(scene_out_dir, f"{name}.png"))
            plt.close()

        # Prepare and plot selected metrics
        def get_vals(key: str):
            return [pf.get(key) for pf in per_frame]

        # SS latent metrics
        plot_series("ss_latent_mean_l2", get_vals("ss_latent_mean_l2"), "SS latent mean L2")
        plot_series(
            "ss_latent_mean_cosine_distance",
            get_vals("ss_latent_mean_cosine_distance"),
            "SS latent mean cosine distance",
        )

        # SLAT metrics
        plot_series("slat_mean_l2", get_vals("slat_mean_l2"), "SLAT mean L2")
        plot_series("slat_mean_cosine_distance", get_vals("slat_mean_cosine_distance"), "SLAT mean cosine distance")
        plot_series("slat_match_ratio_gen", get_vals("slat_match_ratio_gen"), "SLAT match ratio (gen)")
        plot_series("slat_match_ratio_gt", get_vals("slat_match_ratio_gt"), "SLAT match ratio (gt)")

        # Save metrics as JSON for later analysis
        with open(os.path.join(scene_out_dir, f"metrics_{scene_name}.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--first_frame", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=20)
    args = parser.parse_args()

    main(args)
