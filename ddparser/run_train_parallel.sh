#!/bin/bash
GPUS=0,1
export FLAGS_fraction_of_gpu_memory_to_use=0.99
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=True
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=$GPUS
export LD_LIBRARY_PATH=/home/work/nccl_2.3.5/lib:$LD_LIBRARY_PATH
set -x
python -u -m paddle.distributed.launch  --selected_gpus=$GPUS run.py \
                --mode=train \
                --use_cuda \
                --use_data_parallel \
                --feat=char \
                --model_files=model_files/baidu \
                --encoding_model=lstm \
                --preprocess \
                --train_data_path=data/baidu/train.txt \
                --valid_data_path=data/baidu/dev.txt \
                --test_data_path=data/baidu/test.txt \
                --unk=UNK \
                --buckets=15 \
                --batch_size=1000