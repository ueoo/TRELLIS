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
test_data_root = "/viscam/u/yuegao/TRELLIS_datasets/Growth4D_test"
output_root = "/viscam/u/yuegao/TRELLIS_results/"

# finetune_name = ""
# finetune_name = "-calayers"
finetune_name = "-mlp"
pipeline_path = f"{project_root}/TRELLIS-image-large-finetune-growth4d{finetune_name}"

pipeline = TrellisImageTo4DPipeline.from_pretrained(pipeline_path)
pipeline.cuda()


selected_test_cond_frames = [
    "flowerfinal_0001/030.png",
    "jmfloraorchidstemg_0001/000.png",
    "whitemushroom_0001/001.png",
]


cond_frame_paths = [
    os.path.join(test_data_root, "renders", cond_frame_name) for cond_frame_name in selected_test_cond_frames
]

num_frames = 120

for cond_frame_path in cond_frame_paths:
    # Load an image
    image = Image.open(cond_frame_path)

    # Run the pipeline
    outputs = pipeline.run(image, seed=1, formats=["gaussian"], num_frames=num_frames)

    output_folder = os.path.join(output_root, f"results_growth4d{finetune_name.replace('-', '_')}")
    cond_scene_name = cond_frame_path.split("/")[-2]
    cond_frame_name = cond_frame_path.split("/")[-1].split(".")[0]
    sub_folder = (
        f"sample_gs_{num_frames}_test_scene_{cond_scene_name}_frame_{cond_frame_name}{finetune_name.replace('-', '_')}"
    )
    os.makedirs(os.path.join(output_folder, sub_folder), exist_ok=True)

    input_frame_name = f"cond_scene_{cond_scene_name}_frame_{cond_frame_name}.png"
    shutil.copy(cond_frame_path, os.path.join(output_folder, sub_folder, input_frame_name))

    print(f"num outputs: {len(outputs)}")
    rendered_frames = []
    for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Rendering videos"):
        # select view 0 and 100
        # here the "frames" is the views
        renders = render_utils.render_video(output["gaussian"][0], selected_frames=[0, 100])
        rendered_frames.append(renders["color"][0])
    imageio.mimsave(os.path.join(output_folder, sub_folder, f"{i:03d}.mp4"), rendered_frames, fps=30)
