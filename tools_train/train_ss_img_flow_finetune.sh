{
# export CUDA_VISIBLE_DEVICES=0,1
python train.py \
    --config configs/generation_finetune/ss_flow_img_dit_L_16l8_fp16_finetune_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_img_dit_pretrainedvae_flora125geo \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora125Geo_train_merged \
    --master_port 61902 \

exit 0
}
