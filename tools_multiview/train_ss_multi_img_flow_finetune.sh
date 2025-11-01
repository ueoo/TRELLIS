{
export CUDA_VISIBLE_DEVICES=0,1
python train.py \
    --config configs/generation_multiview/ss_flow_multi_img_dit_L_16l8_fp16_finetune_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_multi_img_dit_pretrainedvae_flora125dense \
    --data_dir /scr2/yuegao/TRELLIS_datasets/Flora125Dense  \
    --master_port 62901 \

exit 0
}
