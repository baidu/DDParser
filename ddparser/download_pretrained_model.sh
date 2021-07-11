#!/usr/bin/env bash
# MODEL PATH
MODEL_FILES_PATH="./model_files"
# MODEL_NAME
MODEL_FILES_NAME=DDParser-ernie-lstm-1.0.6.tar.gz

#get pretrained_char_model
wget --no-check-certificate https://ddparser.bj.bcebos.com/$MODEL_FILES_NAME
if [ ! -d $MODEL_FILES_PATH ]; then
	mkdir $MODEL_FILES_PATH
fi
tar xzf $MODEL_FILES_NAME -C $MODEL_FILES_PATH
rm $MODEL_FILES_NAME