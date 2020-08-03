#!/usr/bin/env bash
model_files_path="./model_files"

#get pretrained_char_model
wget --no-check-certificate https://ddparser.bj.bcebos.com/DDParser-char-0.1.0.tar.gz
if [ ! -d $model_files_path ]; then
	mkdir $model_files_path
fi
tar xzf DDParser-char-0.1.0.tar.gz -C $model_files_path
rm DDParser-char-0.1.0.tar.gz