{
export CUDA_VISIBLE_DEVICES=0,1

python train.py \
    --config configs/vae/ss_vae_conv3d_16l8_fp16_finetune.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_vae_conv3d_16l8_fp16_1node_finetune_growth4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Growth4D_train \
    --master_port 2335 \

exit 0
}
