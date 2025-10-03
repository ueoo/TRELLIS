{
export CUDA_VISIBLE_DEVICES=3
python train.py \
    --config configs/generation_finetune/slat_flow_img_dit_L_64l8p2_fp16_finetune_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_img_dit_pretrainedvae_flora4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora4D_train \
    --master_port 69903 \

exit 0
}
