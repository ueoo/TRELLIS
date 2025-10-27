import os


# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import sys

import imageio

from PIL import Image
from tqdm import trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from trellis.pipelines import TrellisMultiImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils


# /viscam/projects/4d-state-machine/TRELLIS-frompretrained/TRELLIS-image-large-flora125-3d-pretrainedvae-multi-imgflow
project_root = "/viscam/projects/4d-state-machine"
# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline_path = (
    f"{project_root}/TRELLIS-frompretrained/TRELLIS-image-large-flora125-3d-pretrainedvae-multi-imgflow-florar0d4"
)
print(f"Loading pipeline from {pipeline_path}")
pipeline = TrellisMultiImageTo3DPipeline.from_pretrained(pipeline_path)
pipeline.cuda()

start_frame = 1
end_frame = 90


output_folder = f"{project_root}/TRELLIS_results/results_flora125_3d_pretrainedvae_multi_imgflow_florar0d4"
os.makedirs(output_folder, exist_ok=True)

data_folder = "/scr/yuegao/TRELLIS_datasets/Flora125Sparse_train/renders_fixview/"
test_scene_name = "floraf025"

fixview_views = [3, 6, 8]

all_videos = []
for frame in trange(start_frame, end_frame + 1, desc=f"Generating frames for {test_scene_name}"):
    # Load an image
    input_frames = []
    for view in fixview_views:
        input_frame_name = f"{test_scene_name}_{frame:04d}/{view:03d}.png"
        image = Image.open(f"{data_folder}/{input_frame_name}")
        input_frames.append(image)

    # Run the pipeline
    outputs = pipeline.run(input_frames, seed=1, formats=["gaussian"])
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    video = render_utils.render_video(outputs["gaussian"][0])["color"]
    imageio.mimsave(f"{output_folder}/{test_scene_name}_frame_{frame:03d}.mp4", video, fps=30)
    all_videos.append(video)
    # video = render_utils.render_video(outputs["radiance_field"][0])["color"]
    # imageio.mimsave(f"{output_folder}/Regrowth_sample_rf_{frame:03d}.mp4", video, fps=30)
    # video = render_utils.render_video(outputs["mesh"][0])["normal"]
    # imageio.mimsave(f"{output_folder}/Regrowth_sample_mesh_{frame:03d}.mp4", video, fps=30)

    # GLB files can be extracted from the outputs
    # glb = postprocessing_utils.to_glb(
    #     outputs["gaussian"][0],
    #     outputs["mesh"][0],
    #     # Optional parameters
    #     simplify=0.95,  # Ratio of triangles to remove in the simplification process
    #     texture_size=1024,  # Size of the texture used for the GLB
    # )
    # glb.export(f"{output_folder}/Regrowth_sample_{frame:03d}.glb")

    # # Save Gaussians as PLY files
    # outputs["gaussian"][0].save_ply(f"{output_folder}/Regrowth_sample_{frame:03d}.ply")

views = len(all_videos[0])
for view_idx in trange(views, desc="Saving views"):
    cur_video = [video[view_idx] for video in all_videos]
    imageio.mimsave(f"{output_folder}/{test_scene_name}_view_{view_idx:03d}.mp4", cur_video, fps=30)
