{

dataset_root=/viscam/u/yuegao/TRELLIS_datasets/FloraSimple4D_test

# dataset_root=/scr/yuegao/TRELLIS_datasets/FloraSimple4D_train

##### rm -rf $dataset_root

python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

python dataset_toolkits/download.py FloraSimple4D --output_dir $dataset_root

python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

#### we already have the rendered images and transforms.json
python dataset_toolkits/render.py FloraSimple4D --output_dir $dataset_root --radius 1.5 --num_views 1

python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

python dataset_toolkits/render_cond.py FloraSimple4D --output_dir $dataset_root --num_views 1

python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

# python dataset_toolkits/voxelize.py FloraSimple4D --output_dir $dataset_root

# python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

# python dataset_toolkits/extract_feature.py --output_dir $dataset_root

# python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

# python dataset_toolkits/encode_ss_latent.py --output_dir $dataset_root

# python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

# python dataset_toolkits/encode_ss_latent.py \
#     --output_dir $dataset_root \
#     --model_root /viscam/projects/4d-state-machine/TRELLIS_outputs \
#     --enc_model ss_vae_conv3d_16l8_fp16_1node_finetune_FloraSimple4D \
#     --ckpt ema0.9999_step0500000 \

# python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

# python dataset_toolkits/render_cond_test.py FloraSimple4D --output_dir $dataset_root

# python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

# python dataset_toolkits/encode_latent.py --output_dir $dataset_root

# python dataset_toolkits/encode_latent.py \
#     --output_dir $dataset_root \
#     --model_root /viscam/projects/4d-state-machine/TRELLIS_outputs \
#     --enc_model slat_vae_enc_dec_gs_swin8_B_64l8_fp16_1node_finetune_FloraSimple4D \
#     --ckpt ema0.9999_step0200000 \


python dataset_toolkits/build_metadata.py FloraSimple4D --output_dir $dataset_root

exit 0
}
