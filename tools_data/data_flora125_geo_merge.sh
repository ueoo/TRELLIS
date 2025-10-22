{

scr_root=/scr/yuegao

dataset_root=$scr_root/TRELLIS_datasets

data_name=Flora125Geo_train_merged
data_dir=$dataset_root/${data_name}


python dataset_toolkits/build_metadata_merge.py --output_dir $data_dir

exit 0
}
