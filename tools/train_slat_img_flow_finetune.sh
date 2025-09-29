{
export CUDA_VISIBLE_DEVICES=4
python train.py \
    --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_img_dit_L_64l8p2_fp16_finetune_growth4d \
    --master_port 24445 \
    --data_dir /scr/yuegao/TRELLIS_datasets/Growth4D_train \

exit 0
}
