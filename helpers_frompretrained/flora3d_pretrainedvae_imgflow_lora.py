from common import build_folder


our_finetune_folder = "TRELLIS-image-large-flora3d-pretrainedvae-imgflow-lora-10k"


# Map model keys to their LoRA training run directories
rename_ops = {
    "sparse_structure_decoder": "copy",
    "sparse_structure_flow_model": "ss_flow_img_all_dit_pretrainedvae_lora_flora4d",
    # "sparse_structure_cond_sparse_structure_flow_model": "ss_flow_ssprev_dit_pretrainedvae_lora_flora4d",
    "slat_decoder_gs": "copy",
    "slat_flow_model": "slat_flow_img_all_dit_pretrainedvae_lora_flora4d",
    # "slat_cond_slat_flow_model": "slat_flow_slatprev_dit_pretrainedvae_lora_flora4d",
}

# Provide only adapter filenames; run dirs come from rename_ops
model_name_to_ckpt = {
    "sparse_structure_flow_model": "denoiser_lora_step0010000.pt",
    # "sparse_structure_cond_sparse_structure_flow_model": "denoiser_lora_step0100000.pt",
    "slat_flow_model": "denoiser_lora_step0010000.pt",
    # "slat_cond_slat_flow_model": "denoiser_lora_step0100000.pt",
}

lora_wrapper_name = {
    "sparse_structure_flow_model": "LoRASparseStructureFlowModel",
    # "sparse_structure_cond_sparse_structure_flow_model": "LoRASparseStructureLatentCondSparseStructureFlowModel",
    "slat_flow_model": "LoRAElasticSLatFlowModel",
    # "slat_cond_slat_flow_model": "LoRAElasticSLatCondSLatFlowModel",
}


build_folder(
    out_folder=our_finetune_folder,
    pipeline_name="TrellisImageTo3DPipeline",
    rename_ops=rename_ops,
    model_name_to_ckpt=model_name_to_ckpt,
    lora_wrapper_name=lora_wrapper_name,
)
