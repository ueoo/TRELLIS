{
export CUDA_VISIBLE_DEVICES=3
python train.py \
    --config configs/generation_finetune/ss_residual_flow_ssprev_dit_L_16l8_fp16_pretrainedvae_final.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_residual_flow_ssprev_dit_L_16l8_fp16_pretrainedvae_final_flora125geo \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora125Geo_train_merged \
    --master_port 61946 \

exit 0
}
