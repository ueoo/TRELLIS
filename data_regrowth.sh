{
# python dataset_toolkits/build_metadata.py Regrowth --output_dir datasets/Regrowth

# python dataset_toolkits/download.py Regrowth --output_dir datasets/Regrowth

# python dataset_toolkits/build_metadata.py Regrowth --output_dir datasets/Regrowth

# python dataset_toolkits/render.py Regrowth --output_dir datasets/Regrowth --radius 1.0 --num_views 150

# python dataset_toolkits/build_metadata.py Regrowth --output_dir datasets/Regrowth

# python dataset_toolkits/voxelize.py Regrowth --output_dir datasets/Regrowth

# python dataset_toolkits/build_metadata.py Regrowth --output_dir datasets/Regrowth

# python dataset_toolkits/extract_feature.py --output_dir datasets/Regrowth

# python dataset_toolkits/build_metadata.py Regrowth --output_dir datasets/Regrowth

# python dataset_toolkits/encode_ss_latent.py \
#     --output_dir datasets/Regrowth \
#     --model_root outputs \
#     --enc_model ss_vae_conv3d_16l8_fp16_1node_finetune_regrowth_debug \
#     --ckpt ema0.9999_step0115000 \

# python dataset_toolkits/build_metadata.py Regrowth --output_dir datasets/Regrowth

python dataset_toolkits/render_cond.py Regrowth --output_dir datasets/Regrowth

# python dataset_toolkits/build_metadata.py Regrowth --output_dir datasets/Regrowth


exit 0
}
