{
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --config configs/generation_finetune/ss_flow_ssprev_dit_L_16l8_fp16_finetune_pretrainedvae_mlp.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_ssprev_dit_pretrainedvae_mlp_flora4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora4D_train \
    --master_port 69900 \

exit 0
}
