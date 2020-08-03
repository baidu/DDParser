data_files_path="./data"

#get data
if [ ! -d $data_files_path ]; then
	mkdir $data_files_path
fi
wget --no-check-certificate https://ddparser.bj.bcebos.com/DDParser_dataset-1.0.0.tar.gz
tar xzf DDParser_dataset-1.0.0.tar.gz -C $data_files_path
rm DDParser_dataset-1.0.0.tar.gz