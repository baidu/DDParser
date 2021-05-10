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
"""
本文件定义BiLSTM网络结构和相关函数
"""

import numpy as np
import paddle
import time
from paddle.fluid import dygraph
from paddle.fluid import initializer
from paddle.fluid import layers

from ddparser.parser.nets import nn
from ddparser.parser.nets import rnn
from ddparser.parser.nets import SharedDropout


class BiLSTM(dygraph.Layer):
    """
    BiLSTM
    TODO：
        Replace this class with the official implementation.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = paddle.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout,
                                   direction="bidirectional")

    def forward(self, x, seq_mask):
        """Forward network"""
        seq_lens = nn.reduce_sum(seq_mask, -1)
        y, _ = self.lstm(x, sequence_length=seq_lens)

        return y
