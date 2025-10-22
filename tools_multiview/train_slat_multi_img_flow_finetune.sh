{
export CUDA_VISIBLE_DEVICES=4,5,6,7
python train.py \
    --config configs/generation_multiview/slat_flow_multi_img_dit_L_64l8p2_fp16_finetune_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/slat_flow_multi_img_dit_pretrainedvae_objaversexl_flora125dense_florar0d4 \
    --data_dir /scr/yuegao/TRELLIS_datasets/ObjaverseXL_sketchfab_Flora125Dense  \
    --master_port 62905 \

exit 0
}
