{
scr_root=/scr/yuegao

dataset_root=$scr_root/TRELLIS_datasets

dataset_name=Flora4D
data_name=Flora125Geo_train
data_dir=$dataset_root/${data_name}_s11

split=sparse_train

rank=0
world_size=9

available_gpus=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)
if [ -z "$available_gpus" ] || [ "$available_gpus" -le 0 ]; then
  available_gpus=1
fi
process_per_gpu=1
gpu_num=$((available_gpus * process_per_gpu))


####### rm -rf $data_dir

# python dataset_toolkits/build_metadata.py $dataset_name --split $split --output_dir $data_dir

# python dataset_toolkits/download.py $dataset_name --split $split --output_dir $data_dir --rank $rank --world_size $world_size

# python dataset_toolkits/build_metadata.py $dataset_name --output_dir $data_dir

# python dataset_toolkits/render_mp.py $dataset_name --output_dir $data_dir --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $dataset_name --output_dir $data_dir

# python dataset_toolkits/render_fixview.py $dataset_name --output_dir $data_dir --num_views 0

# python dataset_toolkits/build_metadata.py $dataset_name --output_dir $data_dir

# python dataset_toolkits/voxelize.py $dataset_name --output_dir $data_dir --rank $rank --world_size $world_size

# python dataset_toolkits/build_metadata.py $dataset_name --output_dir $data_dir

# python dataset_toolkits/extract_feature_mp.py --output_dir $data_dir  --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $dataset_name --output_dir $data_dir

# python dataset_toolkits/encode_ss_latent_mp.py --output_dir $data_dir --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $dataset_name --output_dir $data_dir

# python dataset_toolkits/encode_latent_mp.py --output_dir $data_dir --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $dataset_name --output_dir $data_dir

# python dataset_toolkits/render_cond_mp.py $dataset_name --output_dir $data_dir --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $dataset_name --output_dir $data_dir

exit 0
}
