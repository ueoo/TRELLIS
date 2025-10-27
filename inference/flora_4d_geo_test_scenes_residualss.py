import argparse
import os
import shutil


# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import sys

import imageio

from PIL import Image
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from trellis.pipelines import TrellisImageTo4DPipeline
from trellis.utils import postprocessing_utils, render_utils


def main(args):

    # Load a pipeline from a model folder or a Hugging Face model hub.
    project_root = "/viscam/projects/4d-state-machine"
    results_root = "/viscam/projects/4d-state-machine/TRELLIS_results"

    finetune_name = "flora4dgeo-pretrainedvae-fullflow-residualss"

    pipeline_path = f"{project_root}/TRELLIS-frompretrained/TRELLIS-image-large-{finetune_name}"

    print(f"Loading pipeline from {pipeline_path}")
    pipeline = TrellisImageTo4DPipeline.from_pretrained(pipeline_path)
    pipeline.cuda()

    output_folder = os.path.join(results_root, f"results_{finetune_name.replace('-', '_')}")
    print(f"Saving outputs to {output_folder}")

    test_data_root = "/scr/yuegao/TRELLIS_datasets/Flora125Geo_test_merged"

    test_scene_names = [
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
        start = len(test_scene_names) * args.rank // args.world_size
        end = len(test_scene_names) * (args.rank + 1) // args.world_size
        test_scene_names = test_scene_names[start:end]
    # Further split across GPUs
    test_scene_names = test_scene_names[args.gpu_idx :: args.gpu_num]

    num_frames = args.num_frames

    for scene_name in tqdm(
        test_scene_names,
        total=len(test_scene_names),
        desc=f"GPU {args.gpu_idx}",
        position=int(args.gpu_idx),
        leave=True,
    ):
        # Load an image
        cond_frame_path = os.path.join(
            test_data_root, "renders_fixview", f"{scene_name}_{args.first_frame:04d}", "000.png"
        )

        image = Image.open(cond_frame_path)

        # Run the pipeline
        outputs = pipeline.run(image, seed=1, formats=["gaussian"], num_frames=num_frames)

        sub_folder = f"sample_gs_{num_frames}_test_scene_{scene_name}_frame_{args.first_frame:04d}"
        os.makedirs(os.path.join(output_folder, sub_folder), exist_ok=True)

        input_frame_name = f"input_scene_{scene_name}_frame_{args.first_frame:04d}.png"
        shutil.copy(cond_frame_path, os.path.join(output_folder, sub_folder, input_frame_name))

        print(f"num output frames: {len(outputs)}")

        rendered_videos = []
        for frame_idx, output in tqdm(enumerate(outputs), total=len(outputs), desc="Rendering videos", leave=False):
            video = render_utils.render_video(output["gaussian"][0])["color"]
            rendered_videos.append(video)
            imageio.mimsave(
                os.path.join(output_folder, sub_folder, f"frame_{frame_idx:03d}.mp4"),
                video,
                fps=30,
            )

        num_views = len(rendered_videos[0])
        for view_idx in trange(num_views, desc="Saving views", leave=False):
            cur_view_videos = [rendered_videos[frame_idx][view_idx] for frame_idx in range(len(rendered_videos))]
            imageio.mimsave(
                os.path.join(output_folder, sub_folder, f"view_{view_idx:03d}.mp4"),
                cur_view_videos,
                fps=30,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--first_frame", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=30)
    args = parser.parse_args()

    main(args)
