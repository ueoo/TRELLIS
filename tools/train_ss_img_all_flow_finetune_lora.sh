{
export CUDA_VISIBLE_DEVICES=2
python train.py \
    --config configs/generation_finetune/ss_flow_img_all_dit_L_16l8_fp16_finetune_pretrainedvae_lora.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_img_all_dit_pretrainedvae_lora_flora4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora4D_train \
    --master_port 69902 \

exit 0
}
