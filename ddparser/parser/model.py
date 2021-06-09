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
from functools import reduce

import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid import initializer
from paddle.fluid import layers
from paddle.static import InputSpec
import paddle.inference as paddle_infer

from ddparser.parser.config import ArgConfig
from ddparser.parser.data_struct import utils
from ddparser.parser.data_struct import Embedding
from ddparser.parser.data_struct import Metric
from ddparser.parser.data_struct import MetricInfer
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
        if args.encoding_model == "ernie-lstm":
            self.embed = LSTMByWPEmbed(args)
        else:
            raise KeyError("error encoding_model({args.encoding_model}) should be ernie-lstm.")
        mlp_input_size = self.embed.mlp_input_size

        # mlp layer
        self.mlp_arc_h = MLP(n_in=mlp_input_size, n_out=args.n_mlp_arc, dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=mlp_input_size, n_out=args.n_mlp_arc, dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=mlp_input_size, n_out=args.n_mlp_rel, dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=mlp_input_size, n_out=args.n_mlp_rel, dropout=args.mlp_dropout)

        # biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel, n_out=args.n_rels, bias_x=True, bias_y=True)

    def forward(self, words, feats, batch_size, max_len, token_num):
        """Forward network"""
        # batch_size, seq_len = words.shape
        # get embedding
        words, x = self.embed(words, feats, batch_size, max_len, token_num)

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
        s_arc = layers.elementwise_add(
            s_arc * paddle.cast(s_arc_mask, 'float32'),
            paddle.scale(paddle.cast(s_arc_mask, 'float32'), scale=1e5, bias=-1, bias_after_scale=False))

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
            words, position, arcs, rels = inputs
            batch_size = np.array([words.shape[0]], dtype='int32')
            max_len = np.array([words.shape[1]], dtype='int32')
            token_num = np.array([position.shape[1]], dtype='int32')
            s_arc, s_rel, words = model(words, position, batch_size, max_len, token_num)
        else:
            logging.FATAL("Localzation branch only supports ernie-lstm!")

        mask = layers.logical_and(
            layers.logical_and(words != pad_index, words != bos_index),
            words != eos_index,
        )

        loss = loss_function(s_arc, s_rel, arcs, rels, mask)
        if args.use_data_parallel:
            loss.backward()
        else:
            loss.backward()
        optimizer.minimize(loss)
        total_loss += loss.numpy().item()
        logging.info("epoch: {}, batch: {}/{}, batch_size: {}, loss: {:.4f}".format(
            epoch, batch, math.ceil(len(loader) / args.nranks), len(words),
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
            words, position, arcs, rels = inputs
            batch_size = np.array([words.shape[0]], dtype='int32')
            max_len = np.array([words.shape[1]], dtype='int32')
            token_num = np.array([position.shape[1]], dtype='int32')
            s_arc, s_rel, words = model(words, position, batch_size, max_len, token_num)
        else:
            raise KeyError("error encoding_model({args.encoding_model}) should be ernie-lstm.")

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
def epoch_evaluate_infer(args, model, loader, puncts, input_handles, output_names):
    """Evaluate in one epoch"""
    total_loss, metric = 0, MetricInfer()
    pad_index = args.pad_index
    bos_index = args.bos_index
    eos_index = args.eos_index

    for batch, inputs in enumerate(loader(), start=1):
        if args.encoding_model.startswith("ernie"):
            words, position, arcs, rels = inputs
            words = words.numpy()
            position = position.numpy()
            arcs = arcs.numpy()
            rels = rels.numpy()
            batch_size = np.array([words.shape[0]], dtype='int32')
            max_len = np.array([words.shape[1]], dtype='int32')
            token_num = np.array([position.shape[1]], dtype='int32')
            inputs = [words, position, batch_size, max_len, token_num]
            for handle, _input in zip(input_handles, inputs):
                handle.reshape(_input.shape)
                handle.copy_from_cpu(_input)
            model.run()
            output_names = model.get_output_names()
            outputs = []
            for output_name in output_names:
                output_handle = model.get_output_handle(output_name)
                outputs.append(output_handle.copy_to_cpu())
            s_arc, s_rel, words = outputs
        else:
            raise KeyError("error encoding_model({args.encoding_model}) should be ernie-lstm.")

        mask = np.logical_and(
            np.logical_and(words != pad_index, words != bos_index),
            words != eos_index,
        )
        lens = np.sum(mask, -1)
        arc_preds, rel_preds = decode_infer(args, s_arc, s_rel, mask, lens)
        
        # ignore all punctuation if not specified
        # if not args.punct:
        #     punct_mask = layers.reduce_all(
        #         layers.expand(layers.unsqueeze(words, -1),
        #                       (1, 1, puncts.shape[0])) != layers.expand(layers.reshape(puncts,
        #                                                                                (1, 1, -1)), words.shape + [1]),
        #         dim=-1)

        #     mask = layers.logical_and(mask, punct_mask)

        metric(arc_preds, rel_preds, arcs, rels, mask)


    return metric


@dygraph.no_grad
def epoch_predict(env, args, model, loader):
    """Predict in one epoch"""
    arcs, rels, probs = [], [], []
    pad_index = args.pad_index
    bos_index = args.bos_index
    eos_index = args.eos_index
    for batch, inputs in enumerate(loader(), start=1):
        if args.encoding_model.startswith("ernie"):
            words, position = inputs[0], inputs[1]
            batch_size = np.array([words.shape[0]], dtype='int32')
            max_len = np.array([words.shape[1]], dtype='int32')
            token_num = np.array([position.shape[1]], dtype='int32')
            s_arc, s_rel, words = model(words, position, batch_size, max_len, token_num)

        else:
            raise KeyError("error encoding_model({args.encoding_model}) should be ernie-lstm.")
        mask = layers.logical_and(
            layers.logical_and(words != pad_index, words != bos_index),
            words != eos_index,
        )
        lens = nn.reduce_sum(paddle.cast(mask, dtype='int32'), -1)
        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        arcs.extend(layers.split(nn.masked_select(arc_preds, mask), lens.numpy().tolist()))
        rels.extend(layers.split(nn.masked_select(rel_preds, mask), lens.numpy().tolist()))
        if args.prob:
            arc_probs = nn.index_sample(layers.softmax(s_arc, -1), layers.unsqueeze(arc_preds, -1))
            probs.extend(
                layers.split(
                    nn.masked_select(layers.squeeze(arc_probs, axes=[-1]), mask),
                    lens.numpy().tolist(),
                ))
    arcs = [seq.numpy().tolist() for seq in arcs]
    rels = [env.REL.vocab[seq.numpy().tolist()] for seq in rels]
    probs = [[round(p, 3) for p in seq.numpy().tolist()] for seq in probs]

    return arcs, rels, probs

@dygraph.no_grad
def epoch_predict_infer(env, args, model, loader, input_handles, output_names):
    """Predict in one epoch"""
    arcs, rels, probs = [], [], []
    pad_index = args.pad_index
    bos_index = args.bos_index
    eos_index = args.eos_index
    for batch, inputs in enumerate(loader(), start=1):
        if args.encoding_model.startswith("ernie"):
            words, position = inputs[0].numpy(), inputs[1].numpy()
            batch_size = np.array([words.shape[0]], dtype='int32')
            max_len = np.array([words.shape[1]], dtype='int32')
            token_num = np.array([position.shape[1]], dtype='int32')
            inputs = [words, position, batch_size, max_len, token_num]
            for handle, _input in zip(input_handles, inputs):
                handle.reshape(_input.shape)
                handle.copy_from_cpu(_input)
            model.run()
            output_names = model.get_output_names()
            outputs = []
            for output_name in output_names:
                output_handle = model.get_output_handle(output_name)
                outputs.append(output_handle.copy_to_cpu())
            s_arc, s_rel, words = outputs

        else:
            raise KeyError("error encoding_model({args.encoding_model}) should be ernie-lstm.")
        mask = np.logical_and(
            np.logical_and(words != pad_index, words != bos_index),
            words != eos_index,
        )
        lens = np.sum(mask, -1)
        arc_preds, rel_preds = decode_infer(args, s_arc, s_rel, mask, lens)
        arcs.extend(np.split(arc_preds[mask], np.cumsum(lens))[:-1])
        rels.extend(np.split(rel_preds[mask], np.cumsum(lens))[:-1])
        def softmax(x, axis=-1):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
        if args.prob:
            arc_probs = softmax(s_arc, -1)
            arc_index = np.expand_dims(arc_preds, -1)
            addend = np.reshape(np.arange(0, stop=reduce(lambda x, y : x * y, arc_probs.shape), step=arc_probs.shape[-1]), arc_index.shape)
            arc_probs = np.take(arc_probs, addend + arc_index)
            probs.extend(np.split(np.squeeze(arc_probs, axis=-1)[mask], np.cumsum(lens))[:-1])
    arcs = [seq.tolist() for seq in arcs]
    rels = [env.REL.vocab[seq.tolist()] for seq in rels]
    probs = [[round(p, 3) for p in seq.tolist()] for seq in probs]

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

def decode_infer(args, s_arc, s_rel, mask, lens):
    """Decode function"""
    # prevent self-loops
    arc_preds = np.argmax(s_arc, -1)
    bad = [not utils.istree(seq[:i + 1]) for i, seq in zip(lens, arc_preds)]
    if args.tree and any(bad):
        arc_preds[bad] = utils.eisner(s_arc[bad], mask[bad])
    rel_preds = np.argmax(s_rel, axis=-1)
    # batch_size, seq_len, _ = rel_preds.shape
    rel_index = np.expand_dims(arc_preds, -1)
    addend = np.reshape(np.arange(0, stop=reduce(lambda x, y : x * y, rel_preds.shape), step=rel_preds.shape[-1]), rel_index.shape)
    rel_preds = np.take(rel_preds, addend + rel_index)
    rel_preds = np.squeeze(rel_preds, axis=-1)
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


def load_static(path, use_cuda):
    """Loading model in static graph"""
    # model = paddle.jit.load(path)
    config = paddle_infer.Config(path + ".pdmodel", path + ".pdiparams")
    config.enable_memory_optim()
    if use_cuda:
        config.enable_use_gpu(1000, 0)
    predictor = paddle_infer.create_predictor(config)
    input_names = predictor.get_input_names()
    input_handles = []
    for input_name in input_names:
        input_handles.append(predictor.get_input_handle(input_name))
    output_names = predictor.get_output_names()
    return predictor, input_handles, output_names


def save_static(path, model, args, fields):
    """Saving model in static graph"""
    model = paddle.jit.to_static(model,
                                 input_spec=[
                                     paddle.static.InputSpec(shape=[None, 200], dtype='int64', name='words'),
                                     paddle.static.InputSpec(shape=[None, 200], dtype='int64', name='position'),
                                     paddle.static.InputSpec(shape=[1], dtype='int32', name='batch_size'),
                                     paddle.static.InputSpec(shape=[1], dtype='int32', name='max_len'),
                                     paddle.static.InputSpec(shape=[1], dtype='int32', name='token_num')
                                 ])
    model_path = os.path.join(path, 'model')
    fileds_path = os.path.join(path, 'fields')
    paddle.jit.save(model, model_path)
    with open(model_path + ".args", "wb") as f:
        pickle.dump(args.namespace, f, protocol=2)
    with open(fileds_path, "wb") as f:
        pickle.dump(fields, f, protocol=2)
