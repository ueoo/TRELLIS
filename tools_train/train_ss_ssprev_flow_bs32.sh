{
# export CUDA_VISIBLE_DEVICES=4,5
python train.py \
    --config configs/generation_finetune/ss_flow_ssprev_dit_L_16l8_fp16_pretrainedvae_bs32.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_ssprev_dit_L_16l8_fp16_pretrainedvae_flora125geo \
    --data_dir /scr2/yuegao/TRELLIS_datasets/Flora125Geo_train_merged \
    --master_port 61906 \

exit 0
}
