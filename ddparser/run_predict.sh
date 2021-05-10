#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.99
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=True
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
set -x
python  run.py \
        --mode=predict \
        --use_cuda \
        --encoding_model=ernie-lstm \
        --model_files=model_files/baidu \
        --infer_data_path=data/baidu/test.txt \
        --infer_result_path=data/baidu/test.predict \
        --buckets=15 \
        --tree \