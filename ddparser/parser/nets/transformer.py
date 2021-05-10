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
本文件定义transformer网络
"""

import numpy as np
from paddle.fluid import dygraph
from paddle.fluid import initializer
from paddle.fluid import layers

from ddparser.ernie.modeling_ernie import ErnieModel


class Transformer(dygraph.Layer):
    """
    Transformer
    """
    def __init__(
        self,
        hidden_size,
        vocab_size,
        name,
        num_heads=12,
        num_layers=3,
    ):
        super(Transformer, self).__init__()
        cfg = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": hidden_size,
            "initializer_range": 0.02,
            "max_position_embeddings": 513,
            "num_attention_heads": num_heads,
            "num_hidden_layers": num_layers,
            "type_vocab_size": 2,
            "vocab_size": vocab_size
        }
        self.transformer = ErnieModel(cfg, name=name)

    def forward(self,
                src_ids,
                word_emb=None,
                sent_ids=None,
                pos_ids=None,
                input_mask=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
        """Forward network"""
        return self.transformer(src_ids, word_emb, sent_ids, pos_ids, input_mask, attn_bias, past_cache,
                                use_causal_mask)
