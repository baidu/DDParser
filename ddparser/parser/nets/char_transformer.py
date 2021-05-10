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
"""本文件定义CharTransformer网络"""

import numpy as np
from paddle.fluid import dygraph
from paddle.fluid import layers

from ddparser.parser.nets import nn
from ddparser.parser.nets import BiLSTM
from ddparser.parser.nets import Transformer


class CharTransformer(dygraph.Layer):
    """CharTransformer"""
    def __init__(self, n_chars, n_out, pad_index, nums_heads=12, num_layers=2, name="char_transformer"):
        super(CharTransformer, self).__init__()
        self.n_chars = n_chars
        self.n_out = n_out
        self.pad_index = pad_index

        self.transformer = Transformer(hidden_size=n_out,
                                       vocab_size=n_chars,
                                       num_heads=nums_heads,
                                       num_layers=num_layers,
                                       name=name)

    def forward(self, x):
        """Forward network"""
        mask = layers.reduce_any(x != self.pad_index, -1)
        lens = nn.reduce_sum(mask, -1)
        masked_x = nn.masked_select(x, mask)
        h, _ = self.transformer(masked_x)
        feat_embed = nn.pad_sequence_paddle(layers.split(h, lens.numpy().tolist(), dim=0), self.pad_index)
        return feat_embed
