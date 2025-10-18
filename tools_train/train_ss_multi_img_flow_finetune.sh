{
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py \
    --config configs/generation_finetune/ss_flow_multi_img_dit_L_16l8_fp16_finetune_pretrainedvae.json \
    --output_dir /viscam/projects/4d-state-machine/TRELLIS_outputs/ss_flow_multi_img_dit_pretrainedvae_objaversexl_flora125dense \
    --data_dir /scr/yuegao/TRELLIS_datasets/ObjaverseXL_sketchfab_Flora125Dense  \
    --master_port 62901 \

exit 0
}
