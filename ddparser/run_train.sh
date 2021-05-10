#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.99
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=True
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
set -x
python -u run.py \
        --mode=train \
        --use_cuda \
        --feat=none \
        --preprocess \
        --model_files=model_files/baidu \
        --train_data_path=data/baidu/train.txt \
        --valid_data_path=data/baidu/dev.txt \
        --test_data_path=data/baidu/test.txt \
        --encoding_model=ernie-lstm \
        --buckets=15 
