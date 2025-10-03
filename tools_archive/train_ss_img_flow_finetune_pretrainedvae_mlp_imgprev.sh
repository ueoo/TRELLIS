{
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --config configs/generation_finetune/ss_flow_img_dit_L_16l8_fp16_finetune_pretrainedvae_mlp_imgprev.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_img_dit_L_16l8_fp16_finetune_pretrainedvae_mlp_imgprev \
    --data_dir /scr/yuegao/TRELLIS_datasets/Growth4D_train \
    --master_port 23536 \

exit 0
}
