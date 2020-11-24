#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.99
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=True
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
set -x
python  run.py \
        --mode=evaluate \
        --use_cuda \
        --model_files=model_files/baidu \
        --encoding_model=lstm \
        --test_data_path=data/baidu/test.txt \
        --buckets=15 \
        --tree

