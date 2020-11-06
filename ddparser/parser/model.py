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
本文件定义模型的网络及相关函数
"""

import logging
import math
import os
import pickle
import time

import numpy as np
from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid import initializer
from paddle.fluid import layers

from ddparser.parser.data_struct import utils
from ddparser.parser.data_struct import Embedding
from ddparser.parser.data_struct import Metric
from ddparser.parser.nets import nn
from ddparser.parser.nets import Biaffine
from ddparser.parser.nets import BiLSTM
from ddparser.parser.nets import CharLSTM
from ddparser.parser.nets import CharTransformer
from ddparser.parser.nets import IndependentDropout
from ddparser.parser.nets import MLP
from ddparser.parser.nets import SharedDropout
from ddparser.parser.nets import Transformer


class Model(dygraph.Layer):
    """"Model"""
    def __init__(self, args, pretrained_embed=None):
        super(Model, self).__init__()
        self.args = args
        # the embedding layer
        self.word_embed = dygraph.Embedding(size=(args.n_words, args.n_embed))

        if args.pretrained_embed_shape is not None:
            if pretrained_embed is not None:
                pre_param_attrs = fluid.ParamAttr(
                    name="pretrained_emb",
                    initializer=initializer.NumpyArrayInitializer(
                        pretrained_embed),
                    trainable=True)
                self.pretrained = dygraph.Embedding(
                    size=args.pretrained_embed_shape,
                    param_attr=pre_param_attrs)
                self.word_embed.weight = layers.create_parameter(
                    shape=(self.args.n_words, self.args.n_embed),
                    dtype='float32',
                    default_initializer=initializer.Constant(value=0.0))
            else:
                self.pretrained = dygraph.Embedding(
                    size=args.pretrained_embed_shape)
        # Initialize feat feature, feat can be char or pos
        if args.feat == 'char':
            if args.encoding_model == "lstm":
                self.feat_embed = CharLSTM(n_chars=args.n_feats,
                                           n_embed=args.n_char_embed,
                                           n_out=args.n_lstm_feat_embed,
                                           pad_index=args.feat_pad_index)
            else:
                self.feat_embed = CharTransformer(
                    n_chars=args.n_feats,
                    n_out=args.n_tran_feat_embed,
                    pad_index=args.feat_pad_index,
                    nums_heads=args.n_tran_feat_head,
                    num_layers=args.n_tran_feat_layer)

        else:
            self.feat_embed = dygraph.Embedding(size=(args.n_feats,
                                                      args.n_feat_embed))
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)
        if args.encoding_model == "lstm":
            # lstm layer
            self.lstm = BiLSTM(input_size=args.n_embed +
                               args.n_lstm_feat_embed,
                               hidden_size=args.n_lstm_hidden,
                               num_layers=args.n_lstm_layers,
                               dropout=args.lstm_dropout)
            self.lstm_dropout = SharedDropout(p=args.lstm_dropout)
            mlp_input_size = args.n_lstm_hidden * 2
        else:
            self.transformer = Transformer(hidden_size=args.n_embed +
                                           args.n_tran_feat_embed,
                                           vocab_size=args.n_words,
                                           name="word_transformer",
                                           num_heads=args.n_tran_word_head,
                                           num_layers=args.n_tran_word_layer)
            mlp_input_size = args.n_embed + args.n_tran_feat_embed

        # mlp layer
        self.mlp_arc_h = MLP(n_in=mlp_input_size,
                             n_out=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=mlp_input_size,
                             n_out=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=mlp_input_size,
                             n_out=args.n_mlp_rel,
                             dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=mlp_input_size,
                             n_out=args.n_mlp_rel,
                             dropout=args.mlp_dropout)

        # biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
                                 n_out=args.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def forward(self, words, feats):
        """Forward network"""
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words != self.pad_index
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words >= self.word_embed.weight.shape[0]
            ext_words = nn.mask_fill(words, ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        # embed.size = (batch, seq_len, n_embed * 2)
        embed = layers.concat((word_embed, feat_embed), axis=-1)

        if self.args.encoding_model == "lstm":
            x, _ = self.lstm(embed, mask, self.pad_index)
            x = self.lstm_dropout(x)
        else:
            _, x = self.transformer(words, word_emb=embed)

        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = layers.transpose(self.rel_attn(rel_d, rel_h),
                                 perm=(0, 2, 3, 1))
        # set the scores that exceed the length of each sentence to -1e5
        s_arc_mask = nn.unsqueeze(layers.logical_not(mask), 1)
        s_arc = nn.mask_fill(s_arc, s_arc_mask, -1e5)
        return s_arc, s_rel


def epoch_train(args, model, optimizer, loader, epoch):
    """Train in one epoch"""
    model.train()
    total_loss = 0
    for batch, (words, feats, arcs, rels) in enumerate(loader(), start=1):
        model.clear_gradients()
        # ignore the first token of each sentence
        tmp_words = layers.pad(words[:, 1:],
                               paddings=[0, 0, 1, 0],
                               pad_value=args.pad_index)
        mask = tmp_words != args.pad_index
        s_arc, s_rel = model(words, feats)
        loss = loss_function(s_arc, s_rel, arcs, rels, mask)
        if args.use_data_parallel:
            loss = model.scale_loss(loss)
            loss.backward()
            model.apply_collective_grads()
        else:
            loss.backward()
        optimizer.minimize(loss)

        total_loss += loss.numpy().item()
        logging.info(
            f"epoch: {epoch}, batch: {batch}/{math.ceil(len(loader) / args.nranks)}, batch_size: {len(words)}, loss: {loss.numpy().item():.4f}"
        )
    total_loss /= len(loader)
    return total_loss


@dygraph.no_grad
def epoch_evaluate(args, model, loader, puncts):
    """Evaluate in one epoch"""
    model.eval()

    total_loss, metric = 0, Metric()

    for words, feats, arcs, rels in loader():
        # ignore the first token of each sentence
        tmp_words = layers.pad(words[:, 1:],
                               paddings=[0, 0, 1, 0],
                               pad_value=args.pad_index)
        mask = tmp_words != args.pad_index

        s_arc, s_rel = model(words, feats)
        loss = loss_function(s_arc, s_rel, arcs, rels, mask)
        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        # ignore all punctuation if not specified
        if not args.punct:
            punct_mask = layers.reduce_all(
                layers.expand(layers.unsqueeze(words, -1),
                              (1, 1, puncts.shape[0])) != layers.expand(
                                  layers.reshape(puncts, (1, 1, -1)),
                                  (*words.shape, 1)),
                dim=-1)
            mask = layers.logical_and(mask, punct_mask)

        metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss += loss.numpy().item()

    total_loss /= len(loader)

    return total_loss, metric


@dygraph.no_grad
def epoch_predict(env, args, model, loader):
    """Predict in one epoch"""
    arcs, rels, probs = [], [], []
    for words, feats in loader():
        # ignore the first token of each sentence
        tmp_words = layers.pad(words[:, 1:],
                               paddings=[0, 0, 1, 0],
                               pad_value=args.pad_index)
        mask = tmp_words != args.pad_index
        lens = nn.reduce_sum(mask, -1)
        s_arc, s_rel = model(words, feats)
        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        arcs.extend(
            layers.split(nn.masked_select(arc_preds, mask),
                         lens.numpy().tolist()))
        rels.extend(
            layers.split(nn.masked_select(rel_preds, mask),
                         lens.numpy().tolist()))
        if args.prob:
            arc_probs = nn.index_sample(layers.softmax(s_arc, -1),
                                        layers.unsqueeze(arc_preds, -1))
            probs.extend(
                layers.split(
                    nn.masked_select(layers.squeeze(arc_probs, axes=[-1]),
                                     mask),
                    lens.numpy().tolist()))
    arcs = [seq.numpy().tolist() for seq in arcs]
    rels = [env.REL.vocab[seq.numpy().tolist()] for seq in rels]
    probs = [[round(p, 3) for p in seq.numpy().tolist()] for seq in probs]

    return arcs, rels, probs


def loss_function(s_arc, s_rel, arcs, rels, mask):
    """Loss function"""
    arcs = nn.masked_select(arcs, mask)
    rels = nn.masked_select(rels, mask)
    s_arc = nn.masked_select(s_arc, mask)
    s_rel = nn.masked_select(s_rel, mask)
    s_rel = nn.index_sample(s_rel, layers.unsqueeze(arcs, 1))
    arc_loss = layers.cross_entropy(layers.softmax(s_arc), arcs)
    rel_loss = layers.cross_entropy(layers.softmax(s_rel), rels)
    loss = layers.reduce_mean(arc_loss + rel_loss)

    return loss


def decode(args, s_arc, s_rel, mask):
    """Decode function"""
    mask = mask.numpy()
    lens = np.sum(mask, -1)
    # prevent self-loops
    arc_preds = layers.argmax(s_arc, -1).numpy()
    bad = [not utils.istree(seq[:i + 1]) for i, seq in zip(lens, arc_preds)]
    if args.tree and any(bad):
        arc_preds[bad] = utils.eisner(s_arc.numpy()[bad], mask[bad])
    arc_preds = dygraph.to_variable(arc_preds, zero_copy=False)
    rel_preds = layers.argmax(s_rel, axis=-1)
    batch_size, seq_len, _ = rel_preds.shape
    rel_preds = nn.index_sample(rel_preds, layers.unsqueeze(arc_preds, -1))
    rel_preds = layers.squeeze(rel_preds, axes=[-1])
    return arc_preds, rel_preds


def save(path, args, model, optimizer):
    """Saving model"""
    fluid.save_dygraph(model.state_dict(), path)
    fluid.save_dygraph(optimizer.state_dict(), path)
    with open(path + ".args", "wb") as f:
        pickle.dump(args, f)


def load(path, model=None):
    """Loading model"""
    if model is None:
        with open(path + ".args", "rb") as f:
            args = pickle.load(f)
        model = Model(args)
    model_state, _ = fluid.load_dygraph(path)
    model.set_dict(model_state)
    return model
