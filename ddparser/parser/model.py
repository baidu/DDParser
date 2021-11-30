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
import time
import paddle
import six
try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3

import numpy as np
from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid import initializer
from paddle.fluid import layers

from ddparser.parser.config import ArgConfig
from ddparser.parser.data_struct import utils
from ddparser.parser.data_struct import Embedding
from ddparser.parser.data_struct import Metric
from ddparser.parser.nets import nn
from ddparser.parser.nets import Biaffine
from ddparser.parser.nets import ErnieEmbed
from ddparser.parser.nets import LSTMEmbed
from ddparser.parser.nets import LSTMByWPEmbed
from ddparser.parser.nets import MLP
from ddparser.parser.nets import TranEmbed


class Model(dygraph.Layer):
    """"Model"""
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        if args.encoding_model == "lstm":
            self.embed = LSTMEmbed(args)
        elif args.encoding_model == "transformer":
            self.embed = TranEmbed(args)
        elif args.encoding_model == "ernie-lstm":
            self.embed = LSTMByWPEmbed(args)
        elif args.encoding_model.startswith("ernie"):
            self.embed = ErnieEmbed(args)
        mlp_input_size = self.embed.mlp_input_size

        # mlp layer
        self.mlp_arc_h = MLP(n_in=mlp_input_size, n_out=args.n_mlp_arc, dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=mlp_input_size, n_out=args.n_mlp_arc, dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=mlp_input_size, n_out=args.n_mlp_rel, dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=mlp_input_size, n_out=args.n_mlp_rel, dropout=args.mlp_dropout)

        # biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel, n_out=args.n_rels, bias_x=True, bias_y=True)

    def forward(self, words, feats=None):
        """Forward network"""
        # batch_size, seq_len = words.shape
        # get embedding
        words, x = self.embed(words, feats)
        mask = layers.logical_and(words != self.args.pad_index, words != self.args.eos_index)

        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = layers.transpose(self.rel_attn(rel_d, rel_h), perm=(0, 2, 3, 1))
        # set the scores that exceed the length of each sentence to -1e5
        s_arc_mask = paddle.unsqueeze(mask, 1)
        s_arc = s_arc * s_arc_mask + paddle.scale(
            paddle.cast(s_arc_mask, 'int32'), scale=1e5, bias=-1, bias_after_scale=False)
   
        return s_arc, s_rel, words


def epoch_train(args, model, optimizer, loader, epoch):
    """Train in one epoch"""
    model.train()
    total_loss = 0
    pad_index = args.pad_index
    bos_index = args.bos_index
    eos_index = args.eos_index

    for batch, inputs in enumerate(loader(), start=1):
        model.clear_gradients()

        if args.encoding_model.startswith("ernie"):
            words, arcs, rels = inputs
            s_arc, s_rel, words = model(words)
        else:
            words, feats, arcs, rels = inputs
            s_arc, s_rel, words = model(words, feats)

        mask = layers.logical_and(
            layers.logical_and(words != pad_index, words != bos_index),
            words != eos_index,
        )

        loss = loss_function(s_arc, s_rel, arcs, rels, mask)
        loss.backward()

        optimizer.minimize(loss)
        total_loss += loss.numpy().item()
        logging.info("epoch: {}, batch: {}/{}, batch_size: {}, loss: {:.4f}".format(
            epoch, batch, math.ceil(len(loader)), len(words),
            loss.numpy().item()))
    total_loss /= len(loader)
    return total_loss


@dygraph.no_grad
def epoch_evaluate(args, model, loader, puncts):
    """Evaluate in one epoch"""
    model.eval()
    total_loss, metric = 0, Metric()
    pad_index = args.pad_index
    bos_index = args.bos_index
    eos_index = args.eos_index

    for batch, inputs in enumerate(loader(), start=1):
        if args.encoding_model.startswith("ernie"):
            words, arcs, rels = inputs
            s_arc, s_rel, words = model(words)
        else:
            words, feats, arcs, rels = inputs
            s_arc, s_rel, words = model(words, feats)
        mask = layers.logical_and(
            layers.logical_and(words != pad_index, words != bos_index),
            words != eos_index,
        )
        loss = loss_function(s_arc, s_rel, arcs, rels, mask)
        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        # ignore all punctuation if not specified
        if not args.punct:
            punct_mask = layers.reduce_all(
                layers.expand(layers.unsqueeze(words, -1),
                              (1, 1, puncts.shape[0])) != layers.expand(layers.reshape(puncts,
                                                                                       (1, 1, -1)), words.shape + [1]),
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
    pad_index = args.pad_index
    bos_index = args.bos_index
    eos_index = args.eos_index
    for batch, inputs in enumerate(loader(), start=1):
        if args.encoding_model.startswith("ernie"):
            words = inputs[0]
            s_arc, s_rel, words = model(words)
        else:
            words, feats = inputs
            s_arc, s_rel, words = model(words, feats)
        mask = layers.logical_and(
            layers.logical_and(words != pad_index, words != bos_index),
            words != eos_index,
        )
        lens = nn.reduce_sum(mask, -1)
        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        arcs.extend(layers.split(nn.masked_select(arc_preds, mask), lens.numpy().tolist()))
        rels.extend(layers.split(nn.masked_select(rel_preds, mask), lens.numpy().tolist()))
        if args.prob:
            arc_probs = nn.index_sample(layers.softmax(s_arc, axis=-1), layers.unsqueeze(arc_preds, -1))
            probs.extend(
                layers.split(
                    nn.masked_select(layers.squeeze(arc_probs, axes=[-1]), mask),
                    lens.numpy().tolist(),
                ))
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
    # batch_size, seq_len, _ = rel_preds.shape
    rel_preds = nn.index_sample(rel_preds, layers.unsqueeze(arc_preds, -1))
    rel_preds = layers.squeeze(rel_preds, axes=[-1])
    return arc_preds, rel_preds


def save(path, args, model, optimizer):
    """Saving model"""
    fluid.save_dygraph(model.state_dict(), path)
    fluid.save_dygraph(optimizer.state_dict(), path)
    with open(path + ".args", "wb") as f:
        pickle.dump(args.namespace, f, protocol=2)


def load(path, model=None, mode="evaluate"):
    """Loading model"""
    if model is None:
        with open(path + ".args", "rb") as f:
            args = ArgConfig(["--None"])
            args.namespace = pickle.load(f)
        model = Model(args)
    model_state, _ = fluid.load_dygraph(path)
    model.set_dict(model_state)
    return model
