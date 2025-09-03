{
python train.py \
    --config configs/vae/slat_vae_dec_rf_swin8_B_64l8_fp16_finetune.json \
    --output_dir outputs/slat_vae_dec_rf_swin8_B_64l8_fp16_1node_finetune_regrowth_debug \
    --data_dir datasets/Regrowth \

exit 0
}

