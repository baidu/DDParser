#!/bin/bash
GPUS=0,1,2,3
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
                --feat=pos \
                --model_files=model_files/ud-en-pos-lstm-nltk \
                --encoding_model=lstm \
                --preprocess \
                --train_data_path=data/baidu/ud-treebanks-v2.7/all_english.conllu_nltk \
                --valid_data_path=data/baidu/ud-treebanks-v2.7/UD_English-EWT_nltk/en_ewt-ud-dev.conllu \
                --test_data_path=data/baidu/ud-treebanks-v2.7/UD_English-EWT_nltk/en_ewt-ud-test.conllu  \
                --buckets=15 \
                --batch_size=20000