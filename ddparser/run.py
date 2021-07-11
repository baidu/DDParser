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
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import sys
import os
import datetime
import logging
import math
import six
import paddle
from six.moves import input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    reload(sys)
    sys.setdefaultencoding('utf8')
except:
    pass
import LAC
import numpy as np
import paddle.distributed as dist
from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid import layers

from ddparser.ernie.optimization import AdamW
from ddparser.ernie.optimization import LinearDecay
from ddparser.parser import epoch_train
from ddparser.parser import epoch_evaluate
from ddparser.parser import epoch_predict
from ddparser.parser import save
from ddparser.parser import load
from ddparser.parser import decode
from ddparser.parser import ArgConfig
from ddparser.parser import Environment
from ddparser.parser import Model
from ddparser.parser.data_struct import Corpus
from ddparser.parser.data_struct import TextDataset
from ddparser.parser.data_struct import batchify
from ddparser.parser.data_struct import Field
from ddparser.parser.data_struct import utils
from ddparser.parser.data_struct import Metric
"""
程序入口，定义了训练，评估，预测等函数
"""


def train(env):
    """Train"""
    args = env.args

    logging.info("loading data.")
    train = Corpus.load(args.train_data_path, env.fields)
    dev = Corpus.load(args.valid_data_path, env.fields)
    test = Corpus.load(args.test_data_path, env.fields)
    logging.info("init dataset.")
    train = TextDataset(train, env.fields, args.buckets)
    dev = TextDataset(dev, env.fields, args.buckets)
    test = TextDataset(test, env.fields, args.buckets)
    logging.info("set the data loaders.")
    train.loader = batchify(train, args.batch_size, args.use_data_parallel, True)
    dev.loader = batchify(dev, args.batch_size)
    test.loader = batchify(test, args.batch_size)

    logging.info("{:6} {:5} sentences, ".format('train:', len(train)) + "{:3} batches, ".format(len(train.loader)) +
                 "{} buckets".format(len(train.buckets)))
    logging.info("{:6} {:5} sentences, ".format('dev:', len(dev)) + "{:3} batches, ".format(len(dev.loader)) +
                 "{} buckets".format(len(dev.buckets)))
    logging.info("{:6} {:5} sentences, ".format('test:', len(test)) + "{:3} batches, ".format(len(test.loader)) +
                 "{} buckets".format(len(test.buckets)))

    logging.info("Create the model")
    model = Model(args)

    # init parallel strategy
    if args.use_data_parallel:
        dist.init_parallel_env()
        model = paddle.DataParallel(model)

    if args.encoding_model.startswith(
            "ernie") and args.encoding_model != "ernie-lstm" or args.encoding_model == 'transformer':
        args['lr'] = args.ernie_lr
    else:
        args['lr'] = args.lstm_lr

    if args.encoding_model.startswith("ernie") and args.encoding_model != "ernie-lstm":
        max_steps = 100 * len(train.loader)
        decay = LinearDecay(args.lr, int(args.warmup_proportion * max_steps), max_steps)
    else:
        decay = dygraph.ExponentialDecay(learning_rate=args.lr, decay_steps=args.decay_steps, decay_rate=args.decay)
        
    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.clip)
    
    if args.encoding_model.startswith("ernie") and args.encoding_model != "ernie-lstm":
        optimizer = AdamW(
            learning_rate=decay,
            parameter_list=model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=grad_clip,
        )
    else:
        optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=decay,
            beta1=args.mu,
            beta2=args.nu,
            epsilon=args.epsilon,
            parameter_list=model.parameters(),
            grad_clip=grad_clip,
        )

    total_time = datetime.timedelta()
    best_e, best_metric = 1, Metric()

    puncts = dygraph.to_variable(env.puncts, zero_copy=False)
    logging.info("start training.")

    for epoch in range(1, args.epochs + 1):
        start = datetime.datetime.now()
        # train one epoch and update the parameter
        logging.info("Epoch {} / {}:".format(epoch, args.epochs))
        epoch_train(args, model, optimizer, train.loader, epoch)
        if args.local_rank == 0:
            loss, dev_metric = epoch_evaluate(args, model, dev.loader, puncts)
            logging.info("{:6} Loss: {:.4f} {}".format('dev:', loss, dev_metric))
            loss, test_metric = epoch_evaluate(args, model, test.loader, puncts)
            logging.info("{:6} Loss: {:.4f} {}".format('test:', loss, test_metric))

            t = datetime.datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric and epoch > args.patience // 10:
                best_e, best_metric = epoch, dev_metric
                save(args.model_path, args, model, optimizer)
                logging.info("{}s elapsed (saved)\n".format(t))
            else:
                logging.info("{}s elapsed\n".format(t))
            total_time += t
            if epoch - best_e >= args.patience:
                break
    if args.local_rank == 0:
        model = load(args.model_path, model)
        loss, metric = epoch_evaluate(args, model, test.loader, puncts)
        logging.info("max score of dev is {:.2%} at epoch {}".format(best_metric.score, best_e))
        logging.info("the score of test at epoch {} is {:.2%}".format(best_e, metric.score))
        logging.info("average time of each epoch is {}s".format(total_time / epoch))
        logging.info("{}s elapsed".format(total_time))


def evaluate(env):
    """Evaluate"""
    args = env.args
    puncts = dygraph.to_variable(env.puncts, zero_copy=False)

    logging.info("Load the dataset")
    evaluates = Corpus.load(args.test_data_path, env.fields)
    dataset = TextDataset(evaluates, env.fields, args.buckets)
    # set the data loader
    dataset.loader = batchify(dataset, args.batch_size)

    logging.info("{} sentences, ".format(len(dataset)) + "{} batches, ".format(len(dataset.loader)) +
                 "{} buckets".format(len(dataset.buckets)))
    logging.info("Load the model")
    model = load(args.model_path)

    logging.info("Evaluate the dataset")
    start = datetime.datetime.now()
    loss, metric = epoch_evaluate(args, model, dataset.loader, puncts)
    total_time = datetime.datetime.now() - start
    logging.info("Loss: {:.4f} {}".format(loss, metric))
    logging.info("{}s elapsed, {:.2f} Sents/s".format(total_time, len(dataset) / total_time.total_seconds()))


def predict(env):
    """Predict"""
    args = env.args

    logging.info("Load the dataset")
    if args.prob:
        env.fields = env.fields._replace(PHEAD=Field("prob"))
    predicts = Corpus.load(args.infer_data_path, env.fields)
    dataset = TextDataset(predicts, [env.WORD, env.FEAT], args.buckets)
    # set the data loader
    dataset.loader = batchify(dataset, args.batch_size)
    logging.info("{} sentences, {} batches".format(len(dataset), len(dataset.loader)))

    logging.info("Load the model")
    model = load(args.model_path)
    model.args = args

    logging.info("Make predictions on the dataset")
    start = datetime.datetime.now()
    model.eval()
    pred_arcs, pred_rels, pred_probs = epoch_predict(env, args, model, dataset.loader)
    total_time = datetime.datetime.now() - start
    # restore the order of sentences in the buckets
    indices = np.argsort(np.array([i for bucket in dataset.buckets.values() for i in bucket]))
    predicts.head = [pred_arcs[i] for i in indices]
    predicts.deprel = [pred_rels[i] for i in indices]
    if args.prob:
        predicts.prob = [pred_probs[i] for i in indices]
    logging.info("Save the predicted result to {}".format(args.infer_result_path))
    predicts.save(args.infer_result_path)
    logging.info("{}s elapsed, {:.2f} Sents/s".format(total_time, len(dataset) / total_time.total_seconds()))


def predict_query(env):
    """Predict one query"""
    args = env.args
    logging.info("Load the model")
    model = load(args.model_path)
    model.eval()
    lac_mode = "seg" if args.feat != "pos" else "lac"
    lac = LAC.LAC(mode=lac_mode)
    if args.prob:
        env.fields = env.fields._replace(PHEAD=Field("prob"))

    while True:
        query = input()
        if isinstance(query, six.text_type):
            pass
        else:
            query = query.decode("utf-8")
        if not query:
            logging.info("quit!")
            return
        if len(query) > 200:
            logging.info("The length of the query should be less than 200！")
            continue
        start = datetime.datetime.now()
        lac_results = lac.run([query])
        predicts = Corpus.load_lac_results(lac_results, env.fields)
        dataset = TextDataset(predicts, [env.WORD, env.FEAT])
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size, use_multiprocess=False, sequential_sampler=True)
        pred_arcs, pred_rels, pred_probs = epoch_predict(env, args, model, dataset.loader)
        predicts.head = pred_arcs
        predicts.deprel = pred_rels
        if args.prob:
            predicts.prob = pred_probs
        predicts._print()
        total_time = datetime.datetime.now() - start
        logging.info("{}s elapsed, {:.2f} Sents/s, {:.2f} ms/Sents".format(
            total_time,
            len(dataset) / total_time.total_seconds(),
            total_time.total_seconds() / len(dataset) * 1000))


class DDParser(object):
    """
    DDParser

    Args:
    use_cuda: BOOL, 是否使用gpu
    tree: BOOL, 是否返回树结构
    prob: BOOL, 是否返回弧的概率
    use_pos: BOOL, 是否返回词性标签(仅parse函数生效)
    model_files_path: str, 模型地址, 为None时下载默认模型
    buckets: BOOL, 是否对样本分桶. 若buckets=True，则会对inputs按长度分桶，处理长度不均匀的输入速度更新快，default=False
    batch_size: INT, 批尺寸, 当buckets为False时，每个batch大小均等于batch_size; 当buckets为True时，每个batch的大小约为'batch_size / 当前桶句子的平均长度'。
                当default=None时，分桶batch_size默认等于1000，不分桶默认等于50。
    encoding_model:指定模型，可以选lstm、transformer、ernie-1.0、ernie-tiny等
    """
    def __init__(
        self,
        use_cuda=False,
        tree=True,
        prob=False,
        use_pos=False,
        model_files_path=None,
        buckets=False,
        batch_size=None,
        encoding_model="ernie-lstm",
    ):
        if model_files_path is None:
            if encoding_model in ["lstm", "transformer", "ernie-1.0", "ernie-tiny", "ernie-lstm"]:
                model_files_path = self._get_abs_path(os.path.join("./model_files/", encoding_model))
            else:
                raise KeyError("Unknown encoding model.")

            if not os.path.exists(model_files_path):
                try:
                    utils.download_model_from_url(model_files_path, encoding_model)
                except Exception as e:
                    logging.error("Failed to download model, please try again")
                    logging.error("error: {}".format(e))
                    raise e

        args = [
            "--model_files={}".format(model_files_path), "--config_path={}".format(self._get_abs_path('config.ini')),
            "--encoding_model={}".format(encoding_model)
        ]

        if use_cuda:
            args.append("--use_cuda")
        if tree:
            args.append("--tree")
        if prob:
            args.append("--prob")
        if batch_size:
            args.append("--batch_size={}".format(batch_size))

        args = ArgConfig(args)
        # Don't instantiate the log handle
        args.log_path = None
        self.env = Environment(args)
        self.args = self.env.args
        paddle.set_device(self.env.place)
        self.model = load(self.args.model_path)
        self.model.eval()
        self.lac = None
        self.use_pos = use_pos
        # buckets=None if not buckets else defaults
        if not buckets:
            self.args.buckets = None
        if args.prob:
            self.env.fields = self.env.fields._replace(PHEAD=Field("prob"))
        if self.use_pos:
            self.env.fields = self.env.fields._replace(CPOS=Field("postag"))
        # set default batch size if batch_size is None and not buckets
        if batch_size is None and not buckets:
            self.args.batch_size = 50

    def parse(self, inputs):
        """
        预测未切词的句子。

        Args:
            x: list(str) | str, 未分词的句子，类型为str或list

        Returns:
            outputs: list, 依存分析结果

        Example:
        >>> ddp = DDParser()
        >>> inputs = "百度是一家高科技公司"
        >>> ddp.parse(inputs)
        [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]

        >>> inputs = ["百度是一家高科技公司", "他送了一本书"]
        >>> ddp.parse(inputs)
        [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']},
         {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]

        >>> ddp = DDParser(prob=True, use_pos=True)
        >>> inputs = "百度是一家高科技公司"
        >>> ddp.parse(inputs)
        [{'word': ['百度', '是', '一家', '高科技', '公司'], 'postag': ['ORG', 'v', 'm', 'n', 'n'],
        'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB'], 'prob': [1.0, 1.0, 1.0, 1.0, 1.0]}]
        """
        if not self.lac:
            self.lac = LAC.LAC(mode="lac" if self.use_pos else "seg", use_cuda=self.args.use_cuda)
        if not inputs:
            return
        if isinstance(inputs, six.string_types):
            inputs = [inputs]
        if all([isinstance(i, six.string_types) and i for i in inputs]):
            lac_results = []
            position = 0
            try:
                inputs = [query if isinstance(query, six.text_type) else query.decode("utf-8") for query in inputs]
            except UnicodeDecodeError:
                logging.warning("encoding only supports UTF-8!")
                return

            while position < len(inputs):
                lac_results += self.lac.run(inputs[position:position + self.args.batch_size])
                position += self.args.batch_size
            predicts = Corpus.load_lac_results(lac_results, self.env.fields)
        else:
            logging.warning("please check the foramt of your inputs.")
            return
        dataset = TextDataset(predicts, [self.env.WORD, self.env.FEAT], self.args.buckets)
        # set the data loader

        dataset.loader = batchify(
            dataset,
            self.args.batch_size,
            use_multiprocess=False,
            sequential_sampler=True if not self.args.buckets else False,
        )
        pred_arcs, pred_rels, pred_probs = epoch_predict(self.env, self.args, self.model, dataset.loader)

        if self.args.buckets:
            indices = np.argsort(np.array([i for bucket in dataset.buckets.values() for i in bucket]))
        else:
            indices = range(len(pred_arcs))
        predicts.head = [pred_arcs[i] for i in indices]
        predicts.deprel = [pred_rels[i] for i in indices]
        if self.args.prob:
            predicts.prob = [pred_probs[i] for i in indices]

        outputs = predicts.get_result()
        return outputs

    def parse_seg(self, inputs):
        """
        预测已切词的句子。

        Args:
            x: list(list(str)), 已分词的句子，类型为list

        Returns:
            outputs: list, 依存分析结果

        Example:
        >>> ddp = DDParser()
        >>> inputs = [['百度', '是', '一家', '高科技', '公司'], ['他', '送', '了', '一本', '书']]
        >>> ddp.parse_seg(inputs)
        [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']},
        {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]


        >>> ddp = DDParser(prob=True)
        >>> inputs = [['百度', '是', '一家', '高科技', '公司']]
        >>> ddp.parse_seg(inputs)
        [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2],
        'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB'], 'prob': [1.0, 1.0, 1.0, 1.0, 1.0]}]
        """
        if not inputs:
            return
        if all([isinstance(i, list) and i and all(i) for i in inputs]):
            predicts = Corpus.load_word_segments(inputs, self.env.fields)
        else:
            logging.warning("please check the foramt of your inputs.")
            return
        dataset = TextDataset(predicts, [self.env.WORD, self.env.FEAT], self.args.buckets)
        # set the data loader
        dataset.loader = batchify(
            dataset,
            self.args.batch_size,
            use_multiprocess=False,
            sequential_sampler=True if not self.args.buckets else False,
        )
        pred_arcs, pred_rels, pred_probs = epoch_predict(self.env, self.args, self.model, dataset.loader)

        if self.args.buckets:
            indices = np.argsort(np.array([i for bucket in dataset.buckets.values() for i in bucket]))
        else:
            indices = range(len(pred_arcs))
        predicts.head = [pred_arcs[i] for i in indices]
        predicts.deprel = [pred_rels[i] for i in indices]
        if self.args.prob:
            predicts.prob = [pred_probs[i] for i in indices]

        outputs = predicts.get_result()
        if outputs[0].get("postag", None):
            for output in outputs:
                del output["postag"]
        return outputs

    def _get_abs_path(self, path):
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))


if __name__ == "__main__":
    logging.info("init arguments.")
    args = ArgConfig()
    
    logging.info("init environment.")
    env = Environment(args)

    logging.info("Override the default configs\n{}".format(env.args))
    logging.info("{}\n{}\n{}\n{}".format(env.WORD, env.FEAT, env.ARC, env.REL))
    logging.info("Set the max num of threads to {}".format(env.args.threads))
    logging.info("Set the seed for generating random numbers to {}".format(env.args.seed))
    logging.info("Run the subcommand in mode {}".format(env.args.mode))

    paddle.set_device(env.place)
    mode = env.args.mode
    if mode == "train":
        train(env)
    elif mode == "evaluate":
        evaluate(env)
    elif mode == "predict":
        predict(env)
    elif mode == "predict_q":
        predict_query(env)
    else:
        logging.error("Unknown task mode: {}.".format(mode))
