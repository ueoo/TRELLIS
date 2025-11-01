{
export CUDA_VISIBLE_DEVICES=2,3
python train.py \
    --config configs/generation_multiview/slat_flow_multi_img_dit_L_64l8p2_fp16_finetune_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_multi_img_dit_pretrainedvae_flora125dense \
    --data_dir /scr2/yuegao/TRELLIS_datasets/Flora125Dense  \
    --master_port 62905 \

exit 0
}
