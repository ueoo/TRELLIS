{

scr_root=/scr/yuegao

dataset_root=$scr_root/TRELLIS_datasets

data_name=ObjaverseXL_sketchfab_Flora125Dense
data_dir=$dataset_root/${data_name}


python dataset_toolkits/build_metadata_merge.py --output_dir $data_dir

exit 0
}
