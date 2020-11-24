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

        self.f_cells = dygraph.LayerList()
        self.b_cells = dygraph.LayerList()
        for _ in range(self.num_layers):
            self.f_cells.append(
                rnn.BasicLSTMUnit(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    param_attr=initializer.Xavier(uniform=False),
                    bias_attr=initializer.ConstantInitializer(value=0.0)))
            self.b_cells.append(
                rnn.BasicLSTMUnit(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    param_attr=initializer.Xavier(uniform=False),
                    bias_attr=initializer.ConstantInitializer(value=0.0)))
            input_size = hidden_size * 2


    def permute_hidden(self, hx, index=None):
        """permute hx by index
        
        Args:
            hx: tuple(h, c)， hidden and cell state.
            index: index

        Returns:
            sorted hx
        """
        if index is None:
            return hx
        h = layers.index_select(hx[0], index, dim=1)
        c = layers.index_select(hx[1], index, dim=1)
        return h, c

    def pack_padded_sequence(self, x, mask, pad_index):
        """
        Packs a padded sequences x.

        Args:
            x: input matrix
            mask: mask matrix
            pad_index: pad_index

        Returns:
            new_x: output
            batch_sizes: sort batch_size by step.
            sorted_indices: The index of x sorted by length

        >>> x
        [
            [5, 6, 7, 0],
            [1, 2, 3, 4],
            [8, 9, 0, 0]
        ]
        >>> mask
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, True, False, False]
        ]
        >>> self.pack_padded_sequence(x, mask, 0)
        [1, 5, 8, 2, 6 ,9 , 3 , 7, 4]
        """
        # sentence length
        mask = layers.cast(mask, 'int64')
        lens = layers.reduce_sum(mask, dim=-1)
        # Sort by sentence length in descending order
        _, sorted_indices = layers.argsort(lens, descending=True)
        sorted_x = layers.index_select(x, sorted_indices)
        sorted_mask = layers.index_select(mask, sorted_indices)
        # transpose
        t_x = layers.transpose(sorted_x, perm=[1, 0, 2])
        t_mask = layers.transpose(sorted_mask, perm=[1, 0])
        # mask_select
        new_x = nn.masked_select(t_x, t_mask)
        # Batch by step
        batch_sizes = layers.reduce_sum(t_mask, -1)
        # remove zero
        batch_sizes = nn.masked_select(batch_sizes, batch_sizes != 0)

        return new_x, batch_sizes.numpy().tolist(), sorted_indices

    def pad_packed_sequence(self, x, batch_sizes, unsorted_indices):
        """Pads a packed sequences."""
        h_size = x.shape[1]
        split_x = layers.split(x, batch_sizes, dim=0)
        max_bs = batch_sizes[0]
        step_embs = []
        for step, cur_bs in enumerate(batch_sizes):
            pad_emb = layers.zeros(shape=(max_bs - cur_bs, h_size),
                                   dtype=x.dtype)
            step_emb = layers.concat(input=(split_x[step], pad_emb))
            step_embs.append(step_emb)
        new_x = layers.stack(step_embs, axis=1)
        new_x = layers.index_select(new_x, unsorted_indices)
        return new_x

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        """single bilstm layer forward network"""
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training and self.dropout > 0:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_bs, bs = len(hx_i[0]), batch_sizes[t]
            if last_bs < bs:
                hx_i = [
                    layers.concat((h, ih[last_bs:bs]))
                    for h, ih in zip(hx_i, hx_0)
                ]
            else:
                if bs < hx_i[0].shape[0]:
                    hx_n.append([hx_i[0][bs:], hx_i[1][bs:]])
                hx_i = [h[:bs] for h in hx_i]
            hx_i = [h for h in cell(x[t], *hx_i)]
            output.append(hx_i[0])
            if self.training and self.dropout > 0:
                hx_i[0] = hx_i[0] * hid_mask[:bs]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [layers.concat(h) for h in zip(*reversed(hx_n))]
        output = layers.concat(output)

        return output, hx_n

    def forward(self, x, seq_mask, pad_index, hx=None):
        """Forward network"""
        x, batch_sizes, sorted_indices = self.pack_padded_sequence(
            x, seq_mask, pad_index)
        _, unsorted_indices = layers.argsort(sorted_indices)
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = layers.zeros(shape=(self.num_layers * 2, batch_size,
                                     self.hidden_size),
                              dtype=x[0].dtype)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sorted_indices)
        h = layers.reshape(h, shape=(self.num_layers, 2, -1, self.hidden_size))
        c = layers.reshape(c, shape=(self.num_layers, 2, -1, self.hidden_size))

        for i in range(self.num_layers):
            x = layers.split(x, batch_sizes, dim=0)
            if self.training and self.dropout > 0:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [j * mask[:len(j)] for j in x]
            x_f, (h_f, c_f) = self.layer_forward(x=x,
                                                 hx=(h[i, 0], c[i, 0]),
                                                 cell=self.f_cells[i],
                                                 batch_sizes=batch_sizes)
            x_b, (h_b, c_b) = self.layer_forward(x=x,
                                                 hx=(h[i, 1], c[i, 1]),
                                                 cell=self.b_cells[i],
                                                 batch_sizes=batch_sizes,
                                                 reverse=True)
            x = layers.concat((x_f, x_b), axis=-1)
            h_n.append(layers.stack((h_f, h_b)))
            c_n.append(layers.stack((c_f, c_b)))
        x = self.pad_packed_sequence(x, batch_sizes, unsorted_indices)
        hx = layers.concat(h_n, axis=0), layers.concat(c_n, axis=0)
        hx = self.permute_hidden(hx, unsorted_indices)

        return x, hx
