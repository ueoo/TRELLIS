{
export CUDA_VISIBLE_DEVICES=1
python train.py \
    --config configs/generation/ss_flow_ssprev_dit_L_16l8_fp16_finetune.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_ssprev_dit_L_16l8_fp16_finetune_growth4d \
    --data_dir /scr/yuegao/TRELLIS_datasets/Growth4D_train \
    --master_port 23435 \

exit 0
}
