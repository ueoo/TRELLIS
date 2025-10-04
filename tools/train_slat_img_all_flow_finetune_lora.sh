{
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --config configs/generation_finetune/slat_flow_img_all_dit_L_64l8p2_fp16_finetune_pretrainedvae_lora.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_img_all_dit_pretrainedvae_lora_flora4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora4D_train \
    --master_port 69900 \

exit 0
}
