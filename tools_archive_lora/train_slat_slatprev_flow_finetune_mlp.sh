{
export CUDA_VISIBLE_DEVICES=5
python train.py \
    --config configs/generation_finetune/slat_flow_slatprev_dit_L_64l8p2_fp16_finetune_pretrainedvae_mlp.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_slatprev_dit_pretrainedvae_mlp_florasimple4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/FloraSimple4D_train \
    --master_port 69905 \

exit 0
}
