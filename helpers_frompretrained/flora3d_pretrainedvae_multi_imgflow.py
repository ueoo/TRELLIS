from common import build_folder


our_finetune_folder = "TRELLIS-image-large-flora125-3d-pretrainedvae-multi-imgflow-florar0d4"


# Map model keys to their LoRA training run directories
rename_ops = {
    "sparse_structure_decoder": "copy",
    "sparse_structure_flow_model": "ss_flow_multi_img_dit_pretrainedvae_objaversexl_flora125dense_florar0d4",
    "slat_decoder_gs": "copy",
    "slat_flow_model": "slat_flow_multi_img_dit_pretrainedvae_objaversexl_flora125dense_florar0d4",
}

# Provide only adapter filenames; run dirs come from rename_ops
model_name_to_ckpt = {
    "sparse_structure_flow_model": "denoiser_step0360000.pt",
    "slat_flow_model": "denoiser_step0410000.pt",
}


build_folder(
    out_folder=our_finetune_folder,
    pipeline_name="TrellisMultiImageTo3DPipeline",
    rename_ops=rename_ops,
    model_name_to_ckpt=model_name_to_ckpt,
)
