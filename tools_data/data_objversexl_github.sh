{

# dataset_root=/viscam/u/yuegao/TRELLIS_datasets/Flora4D_test

dataset_root=/scr/yuegao/TRELLIS_datasets

data_name=ObjaverseXL
data_source=github
data_dir=$dataset_root/${data_name}_${data_source}

#### rm -rf $data_dir

python dataset_toolkits/build_metadata.py $data_name --source $data_source --output_dir $data_dir

python dataset_toolkits/download.py $data_name --output_dir $data_dir

# python dataset_toolkits/build_metadata.py $data_name --output_dir $data_dir

# python dataset_toolkits/render.py $data_name --output_dir $data_dir

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
