{
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --config configs/vae/ss_vae_conv3d_16l8_fp16_finetune.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_vae_flora4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Flora4D_train \
    --master_port 69900 \

exit 0
}
