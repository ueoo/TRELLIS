from common import build_folder


our_finetune_folder = "TRELLIS-image-large-growth4d"

rename_ops = {
    "sparse_structure_decoder": "ss_vae_conv3d_16l8_fp16_1node_finetune_growth4d",
    "sparse_structure_flow_model": "ss_flow_img_dit_L_16l8_fp16_finetune_growth4d",
    "sparse_structure_cond_sparse_structure_flow_model": "ss_flow_ssprev_dit_L_16l8_fp16_finetune_growth4d",
    "slat_decoder_gs": "slat_vae_enc_dec_gs_swin8_B_64l8_fp16_1node_finetune_growth4d",
    "slat_flow_model": "slat_flow_img_dit_L_64l8p2_fp16_finetune_growth4d",
    "slat_cond_slat_flow_model": "slat_flow_slatprev_dit_L_64l8p2_fp16_finetune_growth4d",
}

model_name_to_ckpt = {
    "sparse_structure_decoder": "decoder_ema0.9999_step0500000.pt",
    "sparse_structure_flow_model": "denoiser_ema0.9999_step0160000.pt",
    "sparse_structure_cond_sparse_structure_flow_model": "denoiser_ema0.9999_step0200000.pt",
    "slat_decoder_gs": "decoder_ema0.9999_step0200000.pt",
    "slat_flow_model": "denoiser_ema0.9999_step0285000.pt",
    "slat_cond_slat_flow_model": "denoiser_ema0.9999_step0275000.pt",
}


build_folder(
    out_folder=our_finetune_folder,
    pipeline_name="TrellisImageTo4DPipeline",
    rename_ops=rename_ops,
    model_name_to_ckpt=model_name_to_ckpt,
)
