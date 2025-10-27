{
export CUDA_VISIBLE_DEVICES=6,7
python train.py \
    --config configs/generation_finetune/ss_residual_flow_ssprev_dit_L_16l8_fp16_pretrainedvae_every.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_residual_flow_ssprev_dit_L_16l8_fp16_pretrainedvae_every_flora125geo \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora125Geo_train_merged \
    --master_port 61956 \

exit 0
}
