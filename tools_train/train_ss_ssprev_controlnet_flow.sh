{
export CUDA_VISIBLE_DEVICES=3,4,5
python train.py \
    --config configs/generation_finetune/ss_controlnet_flow_ssprev_dit_L_16l8_fp16_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_controlnet_flow_ssprev_dit_L_16l8_fp16_pretrainedvae_flora125geo \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora125Geo_train_merged \
    --master_port 61936 \

exit 0
}
