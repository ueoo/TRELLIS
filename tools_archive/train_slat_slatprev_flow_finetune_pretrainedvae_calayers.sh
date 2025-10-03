{
export CUDA_VISIBLE_DEVICES=1
python train.py \
    --config configs/generation_finetune/slat_flow_slatprev_dit_L_64l8p2_fp16_finetune_pretrainedvae_calayers.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_slatprev_dit_L_64l8p2_fp16_finetune_pretrainedvae_calayers \
    --master_port 24555 \
    --data_dir /scr/yuegao/TRELLIS_datasets/Growth4D_train \

exit 0
}
