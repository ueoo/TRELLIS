{
export CUDA_VISIBLE_DEVICES=6
python train.py \
    --config configs/generation_finetune/ss_flow_img_all_dit_L_16l8_fp16_finetune_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_img_all_dit_pretrainedvae_flora4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora4D_train \
    --master_port 69906 \

exit 0
}
