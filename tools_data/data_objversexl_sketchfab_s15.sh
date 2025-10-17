{
scr_root=/scr-ssd/yuegao
tmp_dir=$scr_root/tmp
mkdir -p $tmp_dir
export TMPDIR=$tmp_dir
export TMP=$tmp_dir
export TEMP=$tmp_dir



dataset_root=$scr_root/TRELLIS_datasets

data_name=ObjaverseXL
data_source=sketchfab
data_dir=$dataset_root/${data_name}_${data_source}

rank=2
world_size=10

#### auto-detect GPUs and pass gpu_num
available_gpus=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)
if [ -z "$available_gpus" ] || [ "$available_gpus" -le 0 ]; then
  available_gpus=1
fi
process_per_gpu=1
gpu_num=$((available_gpus * process_per_gpu))

# python dataset_toolkits/build_metadata.py $data_name --source $data_source --output_dir $data_dir

# python dataset_toolkits/download.py $data_name --output_dir $data_dir --rank $rank --world_size $world_size

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/render_mp.py $data_name --output_dir $data_dir --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/render_fixview_mp.py $data_name --output_dir $data_dir --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/render_cond.py $data_name --output_dir $data_dir --rank $rank --world_size $world_size

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/voxelize.py $data_name --output_dir $data_dir --rank $rank --world_size $world_size

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/extract_feature_mp.py --output_dir $data_dir --batch_size 2 --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/encode_ss_latent_mp.py --output_dir $data_dir --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

python dataset_toolkits/encode_latent_mp.py --output_dir $data_dir --rank $rank --world_size $world_size --gpu_num $gpu_num

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

exit 0
}
