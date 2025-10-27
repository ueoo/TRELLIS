{
export CUDA_VISIBLE_DEVICES=0,1,2 ##5 #6,7
python train.py \
    --config configs/generation_finetune/slat_flow_slatprev_dit_L_64l8p2_fp16_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_slatprev_dit_L_64l8p2_fp16_pretrainedvae_flora125geo \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora125Geo_train_merged \
    --master_port 61908 \

exit 0
}
