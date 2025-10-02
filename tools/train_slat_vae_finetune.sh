{
export CUDA_VISIBLE_DEVICES=1

python train.py \
    --config configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16_finetune.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_vae_enc_dec_gs_flora4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora4D_train \
    --master_port 69901 \

exit 0
}
