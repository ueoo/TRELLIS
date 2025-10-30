import argparse
import os


# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import sys

import imageio

from PIL import Image
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from trellis.pipelines import TrellisMultiImageTo3DPipeline
from trellis.utils import render_utils


def parse_int_list(csv: str):
    return [int(x) for x in csv.split(",") if x != ""]


def main(args):
    # Project/model setup
    project_root = "/viscam/projects/4d-state-machine"
    results_root = f"{project_root}/TRELLIS_results"

    # Keep the original pretrained path used in this script
    finetune_name = "flora125-3d-pretrainedvae-multi-imgflow-florar0d4"
    pipeline_path = f"{project_root}/TRELLIS-frompretrained/TRELLIS-image-large-{finetune_name}"

    print(f"Loading pipeline from {pipeline_path}")
    pipeline = TrellisMultiImageTo3DPipeline.from_pretrained(pipeline_path)
    pipeline.cuda()

    output_folder = os.path.join(results_root, f"results_{finetune_name.replace('-', '_')}")
    print(f"Saving outputs to {output_folder}")

    data_root = "/scr/yuegao/TRELLIS_datasets/ObjaverseXL_sketchfab_Flora125Dense"
    # Data config
    renders_root = os.path.join(data_root, "renders_fixview")

    scene_names = [
        "000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10",
        "000060a495b381230860ca7315a1b585fabc651cf0833b72b6f481771cca4277",
        "0000a09b3b22da52cd0a55918c511184e5bfe4adc846e7a522bc6b227a1781a3",
        "0000b62fec1d42b484fcf80a248c6d65afebe02e021e495db051d5b47d001a32",
        "0001aa38f9609bf802f55536e9a3b3b1ff4ababe22ea6ae6a3258b793d59ceff",
    ]

    # Partition work across processes and GPUs
    if args.world_size > 1:
        start = len(scene_names) * args.rank // args.world_size
        end = len(scene_names) * (args.rank + 1) // args.world_size
        scene_names = scene_names[start:end]

    scene_names = scene_names[args.gpu_idx :: args.gpu_num]

    fixview_views = [0, 1, 2]

    for scene_name in tqdm(
        scene_names,
        total=len(scene_names),
        desc=f"GPU {args.gpu_idx}",
        position=int(args.gpu_idx),
        leave=True,
    ):
        sub_folder = f"sample_gs_objaversexl_scene_{scene_name}"
        os.makedirs(os.path.join(output_folder, sub_folder), exist_ok=True)

        # Load multi-view images for this frame
        input_frames = []
        for view in fixview_views:
            input_rel = f"{scene_name}/{view:03d}.png"
            image_path = os.path.join(renders_root, input_rel)
            image = Image.open(image_path)
            input_frames.append(image)

        # Run the pipeline with multi-image conditioning
        outputs = pipeline.run(input_frames, seed=1, formats=["gaussian"])

        # Render color video for the generated Gaussian scene
        video = render_utils.render_video(outputs["gaussian"][0])["color"]
        imageio.mimsave(
            os.path.join(output_folder, sub_folder, f"results.mp4"),
            video,
            fps=30,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    main(args)
