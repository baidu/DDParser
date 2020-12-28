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
本文件定义GNN网络
"""

from paddle.fluid import layers
from ddparser.parser.nets import nn


class GraphAttentionLayer(object):
    """GraphAttentionLayer"""
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.a1 = layers.create_parameter(shape=[out_features, 1], dtype='float32')
        self.a2 = layers.create_parameter(shape=[out_features, 1], dtype='float32')

    def forward(self, input, adj):
        """Forward network"""
        h = layers.fc(input, size=self.out_features, num_flatten_dims=2)

        _, N, _ = h.shape
        middle_result1 = layers.expand(layers.matmul(h, self.a1), expand_times=(1, 1, N))
        middle_result2 = layers.transpose(layers.expand(layers.matmul(h, self.a2), expand_times=(1, 1, N)),
                                          perm=[0, 2, 1])
        e = layers.leaky_relu(middle_result1 + middle_result2, self.alpha)
        adj = layers.cast(adj, dtype='int32')
        attention = nn.mask_fill(e, adj == 0.0, -1e9)
        attention = layers.softmax(attention, axis=2)
        attention = layers.dropout(attention, self.dropout)
        h_prime = layers.matmul(attention, h)
        if self.concat:
            return layers.elu(h_prime)
        else:
            return h_prime


class GAT(object):
    """GAT"""
    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, heads, layer):
        self.dropout = dropout
        self.layer = layer
        if self.layer == 1:
            self.attentions = [
                GraphAttentionLayer(input_size, output_size, dropout=dropout, alpha=alpha, concat=True)
                for _ in range(heads)
            ]
        else:
            self.attentions = [
                GraphAttentionLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True)
                for _ in range(heads)
            ]
            self.out_att = GraphAttentionLayer(hidden_size * heads,
                                               output_size,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=False)

    def forward(self, x, adj):
        """Forward network"""

        x = layers.dropout(x, self.dropout)
        if self.layer == 1:
            x = layers.stack([att.forward(x, adj) for att in self.attentions], dim=2)
            x = layers.reduce_sum(x, 2)
            x = layers.dropout(x, self.dropout)
            return layers.log_softmax(x, axis=2)
        else:
            x = layers.concat([att.forward(x, adj) for att in self.attentions], axis=2)
            x = layers.dropout(x, self.dropout)
            return self.out_att.forward(x, adj)
