import os
import sys


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from common import build_folder


our_finetune_folder = "TRELLIS-image-large-flora3dgeo-pretrainedvae-fullflow"

rename_ops = {
    "sparse_structure_decoder": "copy",
    "sparse_structure_flow_model": "ss_flow_img_dit_pretrainedvae_flora125geo",
    # "sparse_structure_cond_sparse_structure_flow_model": "",
    "slat_decoder_gs": "copy",
    "slat_flow_model": "slat_flow_img_dit_pretrainedvae_flora125geo",
    # "slat_cond_slat_flow_model": "",
}

model_name_to_ckpt = {
    "sparse_structure_flow_model": "denoiser_step0480000.pt",
    # "sparse_structure_cond_sparse_structure_flow_model": ".pt",
    "slat_flow_model": "denoiser_step0790000.pt",
    # "slat_cond_slat_flow_model": ".pt",
}

build_folder(
    out_folder=our_finetune_folder,
    pipeline_name="TrellisImageTo3DPipeline",
    rename_ops=rename_ops,
    model_name_to_ckpt=model_name_to_ckpt,
)
