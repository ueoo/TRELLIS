{
export CUDA_VISIBLE_DEVICES=2,3
python train.py \
    --config configs/generation_finetune/slat_flow_img_dit_L_64l8p2_fp16_finetune_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_img_dit_pretrainedvae_flora125geo \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora125Geo_train_merged \
    --master_port 61904 \

exit 0
}
