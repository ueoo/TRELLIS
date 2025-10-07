{
export CUDA_VISIBLE_DEVICES=3
python train.py \
    --config configs/generation_finetune/ss_flow_ssprev_dit_L_16l8_fp16_finetune_pretrainedvae_lora.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_ssprev_dit_pretrainedvae_lora_florasimple4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/FloraSimple4D_train \
    --master_port 69903 \

exit 0
}
