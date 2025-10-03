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


# Load a pipeline from a model folder or a Hugging Face model hub.
project_root = "/viscam/projects/4d-state-machine"
results_root = "/viscam/projects/4d-state-machine/TRELLIS_results"

# finetune_name = "-pretrainedvae-fullflow-ema"
# finetune_name = "-pretrainedvae-fullflow-ema-imgflow10k"
# finetune_name = "-pretrainedvae-prevflow-ema"
# finetune_name = "-pretrainedvae-prevflow-ema-flow50k"
finetune_name = "-pretrainedvae-prevflow-ema-flow10k"

pipeline_path = f"{project_root}/TRELLIS-image-large-finetune-flora4d{finetune_name}"
print(f"Loading pipeline from {pipeline_path}")
pipeline = TrellisImageTo4DPipeline.from_pretrained(pipeline_path)
pipeline.cuda()

test_data_root = "/viscam/u/yuegao/TRELLIS_datasets/Flora4D_test"

selected_test_cond_frames = [
    "jmfloraorchidstemg_0001/000.png",
]

train_data_root = "/scr/yuegao/TRELLIS_datasets/Flora4D_train"
selected_train_cond_frames = [
    "jmfloraorchidstema_0001/000.png",
]

render_folder = "renders"

test_cond_frame_paths = [
    os.path.join(test_data_root, render_folder, cond_frame_name) for cond_frame_name in selected_test_cond_frames
]

train_cond_frame_paths = [
    os.path.join(train_data_root, render_folder, cond_frame_name) for cond_frame_name in selected_train_cond_frames
]

cond_frame_paths = test_cond_frame_paths + train_cond_frame_paths
# cond_frame_paths = train_cond_frame_paths
num_frames = 60

for cond_frame_path in cond_frame_paths:
    # Load an image
    cond_scene_name = cond_frame_path.split("/")[-2]
    cond_frame_name = cond_frame_path.split("/")[-1].split(".")[0]
    split = "test" if cond_frame_path in test_cond_frame_paths else "train"

    print(f"Processing {cond_frame_path}...")

    image = Image.open(cond_frame_path)

    # Run the pipeline
    outputs = pipeline.run(image, seed=1, formats=["gaussian"], num_frames=num_frames)

    output_folder = os.path.join(results_root, f"results_flora4d{finetune_name.replace('-', '_')}_renders")

    sub_folder = f"sample_gs_{num_frames}_{split}_scene_{cond_scene_name}_frame_{cond_frame_name}{finetune_name.replace('-', '_')}"
    os.makedirs(os.path.join(output_folder, sub_folder), exist_ok=True)

    input_frame_name = f"input_scene_{cond_scene_name}_frame_{cond_frame_name}.png"
    shutil.copy(cond_frame_path, os.path.join(output_folder, sub_folder, input_frame_name))

    print(f"num output frames: {len(outputs)}")

    rendered_videos = []
    for frame_idx, output in tqdm(enumerate(outputs), total=len(outputs), desc="Rendering videos"):
        video = render_utils.render_video(output["gaussian"][0])["color"]
        rendered_videos.append(video)
        imageio.mimsave(os.path.join(output_folder, sub_folder, f"frame_{frame_idx:03d}.mp4"), video, fps=30)

    num_views = len(rendered_videos[0])
    for view_idx in trange(num_views, desc="Saving views"):
        cur_view_videos = [rendered_videos[frame_idx][view_idx] for frame_idx in range(len(rendered_videos))]
        imageio.mimsave(os.path.join(output_folder, sub_folder, f"view_{view_idx:03d}.mp4"), cur_view_videos, fps=30)
