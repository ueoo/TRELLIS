{
export CUDA_VISIBLE_DEVICES=6
python train.py \
    --config configs/generation_finetune/ss_flow_ssprev_dit_L_16l8_fp16_finetune_pretrainedvae_mlp.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_ssprev_dit_pretrainedvae_mlp_florasimple4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/FloraSimple4D_train \
    --master_port 69906 \

exit 0
}
