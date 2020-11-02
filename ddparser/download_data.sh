# DATA PATH
DATA_FILES_PATH="./data"
# DATA FILE NAME
DATA_FILES_NAME=DDParser_dataset-1.0.0.tar.gz
#get data
if [ ! -d $DATA_FILES_PATH ]; then
	mkdir $DATA_FILES_PATH
fi
wget --no-check-certificate https://ddparser.bj.bcebos.com/$DATA_FILES_NAME
tar xzf $DATA_FILES_NAME -C $DATA_FILES_PATH
rm $DATA_FILES_NAME