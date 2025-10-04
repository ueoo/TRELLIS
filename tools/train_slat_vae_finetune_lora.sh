{
# export CUDA_VISIBLE_DEVICES=4
python train.py \
    --config configs/vae_finetune/slat_vae_enc_dec_gs_swin8_B_64l8_fp16_finetune_lora.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_vae_enc_dec_gs_flora4d_lora \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora4D_train \
    --master_port 69904 \

exit 0
}
