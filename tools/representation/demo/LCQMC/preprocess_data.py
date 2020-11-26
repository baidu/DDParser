# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
"""preprocess LCQMC data"""
import sys
import json
import pickle
sys.path.append("..")
sys.path.append("../../../..")
from ERNIE.tokenization import BasicTokenizer

import pandas as pd
from ddparser import DDParser

TRAIN_PATH = 'train.csv'
DEV_PATH = 'dev.csv'
TEST_PATH = 'test.csv'
use_cuda = True

if __name__ == "__main__":
    file_paths = [TRAIN_PATH, DEV_PATH, TEST_PATH]
    tokenizer = BasicTokenizer()
    ddp = DDParser(use_cuda=use_cuda,
                   encoding_model='transformer',
                   buckets=True,
                   batch_size=1000)

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\t')
        df['ddp_res_a'] = [
            str(ddp_res) for ddp_res in ddp.parse([
                tokenizer._clean_text(query)
                for query in df['text_a'].tolist()
            ])
        ]
        df['ddp_res_b'] = [
            str(ddp_res) for ddp_res in ddp.parse([
                tokenizer._clean_text(query)
                for query in df['text_b'].tolist()
            ])
        ]
        output_path = file_path.split('.')[0] + '_ddp.csv'
        df.to_csv(output_path, sep='\t', index=False)
        print(f"{file_path} done!")
