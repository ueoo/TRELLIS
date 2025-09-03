import os


# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio

from PIL import Image
from tqdm import tqdm, trange

from trellis.pipelines import TrellisImageTo4DPipeline
from trellis.utils import postprocessing_utils, render_utils


# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo4DPipeline.from_pretrained("/viscam/projects/4d-state-machine/TRELLIS-image-large-finetune")
pipeline.cuda()

first_cond_frame_path = (
    "datasets/Regrowth/renders_cond/268326481a2fab1e603ea9dcfbb500a6106c7d4c3fa84db5c8573b6eb52d7890/007.png"
)

num_frames = 120

# Load an image
image = Image.open(first_cond_frame_path)

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    formats=["gaussian"],
    num_frames=num_frames,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a list of dictionary containing generated 3D assets in different formats:
# - output['gaussian']: a list of 3D Gaussians
# - output['radiance_field']: a list of radiance fields
# - output['mesh']: a list of meshes

output_folder = "results_4d"
sub_folder = f"Regrowth_sample_gs_{num_frames}_cond_time0_frame{first_cond_frame_path.split('/')[-1].split('.')[0]}"
os.makedirs(os.path.join(output_folder, sub_folder), exist_ok=True)

print(f"num outputs: {len(outputs)}")
for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Rendering videos"):
    video = render_utils.render_video(output["gaussian"][0])["color"]
    imageio.mimsave(os.path.join(output_folder, sub_folder, f"{i:03d}.mp4"), video, fps=30)
