{
python train.py \
    --config configs/vae/ss_vae_conv3d_16l8_fp16_finetune.json \
    --output_dir outputs/ss_vae_conv3d_16l8_fp16_1node_finetune_regrowth_debug \
    --data_dir datasets/Regrowth \

exit 0
}
