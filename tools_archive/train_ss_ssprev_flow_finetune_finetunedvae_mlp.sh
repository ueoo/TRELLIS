{
export CUDA_VISIBLE_DEVICES=2
python train.py \
    --config configs/generation_finetune/ss_flow_ssprev_dit_L_16l8_fp16_finetune_finetunedvae_mlp.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_ssprev_dit_L_16l8_fp16_finetune_finetunedvae_mlp_growth4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Growth4D_train \
    --master_port 25436 \

exit 0
}
