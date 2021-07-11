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
本文件定义Embedding类
"""
import json
import os
import paddle

from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid import initializer
from paddle.fluid import layers

from ddparser.ernie.modeling_ernie import ErnieModel
from ddparser.ernie.modeling_ernie import ErnieEmbedding
from ddparser.parser.nets import nn
from ddparser.parser.nets import BiLSTM
from ddparser.parser.nets import CharLSTM
from ddparser.parser.nets import CharTransformer
from ddparser.parser.nets import IndependentDropout
from ddparser.parser.nets import SharedDropout
from ddparser.parser.nets import Transformer


class PretraEmbedding(dygraph.Layer):
    def __init__(self, args):
        super(PretraEmbedding, self).__init__()
        self.args = args
        # the embedding layer
        self.word_embed = dygraph.Embedding(size=(args.n_words, args.n_embed))
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

    def forward(self, words, feats):
        # get outputs from embedding layers
        word_embed = self.word_embed(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        # embed.size = (batch, seq_len, n_embed * 2)
        embed = layers.concat((word_embed, feat_embed), axis=-1)
        return embed


class LSTMEmbed(PretraEmbedding):
    def __init__(self, args):
        super(LSTMEmbed, self).__init__(args)

        # Initialize feat feature, feat can be char or pos
        if args.feat == "char":
            self.feat_embed = CharLSTM(
                n_chars=args.n_feats,
                n_embed=args.n_char_embed,
                n_out=args.n_lstm_feat_embed,
                pad_index=args.feat_pad_index,
            )
            feat_embed_size = args.n_lstm_feat_embed

        else:
            self.feat_embed = dygraph.Embedding(size=(args.n_feats, args.n_feat_embed))
            feat_embed_size = args.n_feat_embed

        # lstm layer
        self.lstm = BiLSTM(
            input_size=args.n_embed + feat_embed_size,
            hidden_size=args.n_lstm_hidden,
            num_layers=args.n_lstm_layers,
            dropout=args.lstm_dropout,
        )
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)
        self.mlp_input_size = args.n_lstm_hidden * 2

    def forward(self, words, feats):
        mask = words != self.args.pad_index
        embed = super(LSTMEmbed, self).forward(words, feats)
        x = self.lstm(embed, mask)
        x = self.lstm_dropout(x)
        return words, x


class TranEmbed(PretraEmbedding):
    def __init__(self, args):
        super(TranEmbed, self).__init__(args)

        # Initialize feat feature, feat can be char or pos
        if args.feat == "char":
            self.feat_embed = CharTransformer(
                n_chars=args.n_feats,
                n_out=args.n_tran_feat_embed,
                pad_index=args.feat_pad_index,
                nums_heads=args.n_tran_feat_head,
                num_layers=args.n_tran_feat_layer,
            )
            feat_embed_size = args.n_tran_feat_embed

        else:
            self.feat_embed = dygraph.Embedding(size=(args.n_feats, args.n_feat_embed))
            feat_embed_size = args.n_feat_embed

        self.transformer = Transformer(
            hidden_size=args.n_embed + feat_embed_size,
            vocab_size=args.n_words,
            name="word_transformer",
            num_heads=args.n_tran_word_head,
            num_layers=args.n_tran_word_layer,
        )
        self.mlp_input_size = args.n_embed + feat_embed_size

    def forward(self, words, feats):
        embed = super().forward(words, feats)
        _, x = self.transformer(words, word_emb=embed)
        return words, x


class ErnieEmbed(dygraph.Layer):
    def __init__(self, args):
        super(ErnieEmbed, self).__init__()
        self.args = args
        self.init_ernie_model(args)
        self.mlp_input_size = self.ernie.cfg["hidden_size"]

    def init_ernie_model(self, args):
        if args.mode == "train":
            self.ernie = ErnieModel.from_pretrained(args.encoding_model)
            args["ernie_config"] = self.ernie.cfg
        else:
            self.ernie = ErnieModel(args.ernie_config)

    def flat_words(self, words):
        pad_index = self.args.pad_index
        lens = nn.reduce_sum(words != pad_index, dim=-1)
        position = layers.cumsum(lens + layers.cast((lens == 0), "int32"), axis=1) - 1
        flat_words = nn.masked_select(words, words != pad_index)
        flat_words = nn.pad_sequence_paddle(
            layers.split(flat_words,
                         layers.reduce_sum(lens, -1).numpy().tolist(), pad_index))
        max_len = flat_words.shape[1]
        position = nn.mask_fill(position, position >= max_len, max_len - 1)
        return flat_words, position

    def forward(self, words, feats):
        words, position = self.flat_words(words)
        _, encoded = self.ernie(words)
        x = layers.reshape(
            nn.index_sample(encoded, position),
            shape=position.shape[:2] + [encoded.shape[2]],
        )
        words = nn.index_sample(words, position)
        return words, x


class LSTMByWPEmbed(PretraEmbedding):
    def __init__(self, args):
        super(LSTMByWPEmbed, self).__init__(args)
        self.args = args
        self.init_ernie_model(args)
        # lstm layer
        self.lstm = BiLSTM(
            input_size=args.lstm_by_wp_embed_size,
            hidden_size=args.n_lstm_hidden,
            num_layers=args.n_lstm_layers,
            dropout=args.lstm_dropout,
        )
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)
        self.mlp_input_size = args.n_lstm_hidden * 2

    def init_ernie_model(self, args):
        self.word_embed = paddle.nn.Embedding(args.ernie_vocabs_size, args.lstm_by_wp_embed_size)

    def flat_words(self, words):
        pad_index = self.args.pad_index
        lens = nn.reduce_sum(words != pad_index, dim=-1)
        position = layers.cumsum(lens + layers.cast((lens == 0), "int32"), axis=1) - 1
        flat_words = nn.masked_select(words, words != pad_index)
        flat_words = nn.pad_sequence_paddle(
            layers.split(flat_words,
                         layers.reduce_sum(lens, -1).numpy().tolist(), pad_index))
        max_len = flat_words.shape[1]
        position = nn.mask_fill(position, position >= max_len, max_len - 1)
        return flat_words, position

    def forward(self, words, feats):
        words, position = self.flat_words(words)
        word_embed = self.word_embed(words)
        # word_embed = self.embed_dropout(word_embed)
        # concatenate the word and feat representations
        # embed.size = (batch, seq_len, n_embed * 2)
        embed = word_embed
        mask = words != self.args.pad_index
        x = self.lstm(embed, mask)
        x = layers.reshape(nn.index_sample(x, position), shape=position.shape[:2] + [x.shape[2]])
        words = paddle.index_sample(words, position)
        x = self.lstm_dropout(x)

        return words, x