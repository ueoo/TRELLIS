{
export CUDA_VISIBLE_DEVICES=2
python train.py \
    --config configs/generation_finetune/slat_flow_img_dit_L_64l8p2_fp16_finetune_pretrainedvae_mlp_imgcondprev.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_img_dit_L_64l8p2_fp16_finetune_pretrainedvae_mlp_imgcondprev \
    --data_dir /scr/yuegao/TRELLIS_datasets/Growth4D_train \
    --master_port 29336 \

exit 0
}
