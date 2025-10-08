{

tmp_dir=/scr/yuegao/tmp
mkdir -p $tmp_dir
export TMPDIR=$tmp_dir
export TMP=$tmp_dir
export TEMP=$tmp_dir

dataset_root=/scr/yuegao/TRELLIS_datasets

data_name=ObjaverseXL
data_source=sketchfab
data_dir=$dataset_root/${data_name}_${data_source}

rank=22
world_size=25

# python dataset_toolkits/build_metadata.py $data_name --source $data_source --output_dir $data_dir

# python dataset_toolkits/download.py $data_name --output_dir $data_dir --rank $rank --world_size $world_size

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

python dataset_toolkits/render.py $data_name --output_dir $data_dir --render_frames --rank $rank --world_size $world_size

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/render_cond.py $data_name --output_dir $data_dir

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/voxelize.py $data_name --output_dir $data_dir

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/extract_feature.py --output_dir $data_dir

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/encode_ss_latent.py --output_dir $data_dir

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir


# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/render_cond_test.py $data_name --output_dir $data_dir

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/encode_latent.py --output_dir $data_dir

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

exit 0
exit 0
}
