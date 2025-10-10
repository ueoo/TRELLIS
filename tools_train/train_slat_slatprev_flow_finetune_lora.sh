{
export CUDA_VISIBLE_DEVICES=1
python train.py \
    --config configs/generation_finetune/slat_flow_slatprev_dit_L_64l8p2_fp16_finetune_pretrainedvae_lora.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_slatprev_dit_pretrainedvae_lora_florasimple4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/FloraSimple4D_train \
    --master_port 69901 \

exit 0
}
