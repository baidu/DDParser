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
本文件初始化配置和环境的相关类
"""

import ast
import argparse
import configparser
import logging
import os
import math
import pickle

import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import dygraph

from ddparser.ernie.tokenizing_ernie import ErnieTokenizer
from ddparser.parser.data_struct import utils
from ddparser.parser.data_struct import CoNLL
from ddparser.parser.data_struct import Corpus
from ddparser.parser.data_struct import Embedding
from ddparser.parser.data_struct import ErnieField
from ddparser.parser.data_struct import Field
from ddparser.parser.data_struct import SubwordField


class ArgumentGroup(object):
    """ArgumentGroup"""
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, *args, **kwargs):
        self._group.add_argument(*args, **kwargs)


class ArgConfig(configparser.ConfigParser):
    """ArgConfig class for reciving parameters"""
    def __init__(self, args=None):
        super(ArgConfig, self).__init__()

        parser = argparse.ArgumentParser(description="BaiDu's Denpendency Parser.")
        model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
        model_g.add_arg(
            "--mode",
            default="train",
            choices=["train", "evaluate", "predict", "predict_q"],
            help="Select task mode",
        )
        model_g.add_arg("--config_path", "-c", default="config.ini", help="path to config file")
        model_g.add_arg(
            "--model_files",
            default="model_files/baidu",
            help="Directory path to save model and ",
        )

        data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
        data_g.add_arg("--train_data_path", help="path to training data.")
        data_g.add_arg("--valid_data_path", help="path to valid data.")
        data_g.add_arg("--test_data_path", help="path to testing data.")
        data_g.add_arg("--infer_data_path", help="path to dataset")
        data_g.add_arg("--batch_size", default=1000, type=int, help="batch size")

        log_g = ArgumentGroup(parser, "logging", "logging related")
        log_g.add_arg("--log_path", default="./log/log", help="log path")
        log_g.add_arg(
            "--log_level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"],
            help="log level",
        )
        log_g.add_arg(
            "--infer_result_path",
            default="infer_result",
            help="Directory path to infer result.",
        )

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg(
            "--use_cuda",
            "-gpu",
            action="store_true",
            help="If set, use GPU for training.",
        )
        run_type_g.add_arg(
            "--preprocess",
            "-p",
            action="store_true",
            help="whether to preprocess the data first",
        )
        run_type_g.add_arg(
            "--use_data_parallel",
            action="store_true",
            help="The flag indicating whether to use data parallel mode to train the model.",
        )
        run_type_g.add_arg(
            "--seed",
            "-s",
            default=1,
            type=int,
            help="seed for generating random numbers",
        )
        run_type_g.add_arg("--threads", "-t", default=16, type=int, help="max num of threads")
        run_type_g.add_arg("--tree", action="store_true", help="whether to ensure well-formedness")
        run_type_g.add_arg("--prob", action="store_true", help="whether to output probs")

        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg(
            "--feat",
            default="none",
            choices=["pos", "char", "none"],
            help="choices of additional features",
        )
        train_g.add_arg(
            "--encoding_model",
            default="ernie-lstm",
            choices=["lstm", "transformer", "ernie-1.0", "ernie-tiny", "ernie-lstm"],
            help="choices of encode model",
        )
        train_g.add_arg("--buckets", default=15, type=int, help="max num of buckets to use")
        train_g.add_arg("--punct", action="store_true", help="whether to include punctuation")

        custom_g = ArgumentGroup(parser, "customize", "customized options.")
        custom_g.add_arg("--None", action="store_true", help="None")

        self.build_conf(parser, args)

    def build_conf(self, parser, args=None):
        """Initialize the parameters, then combine the parameters parsed by the parser and the parameters read by config"""
        args = parser.parse_args(args)
        self.read(args.config_path)
        self.namespace = argparse.Namespace()
        self.update(
            dict((name, ast.literal_eval(value)) for section in self.sections() for name, value in self.items(section)))
        args.nranks = paddle.distributed.get_world_size()
        args.local_rank = paddle.distributed.get_rank()
        args.fields_path = os.path.join(args.model_files, "fields")
        args.model_path = os.path.join(args.model_files, "model")
        # update config from args
        self.update(vars(args))
        return self

    def __repr__(self):
        """repr"""
        s = line = "-" * 25 + "-+-" + "-" * 25 + "\n"
        s += "{:25} | {:^25}\n".format('Param', 'Value') + line
        for name, value in vars(self.namespace).items():
            s += "{:25} | {:^25}\n".format(name, str(value))
        s += line

        return s

    def __getattr__(self, attr):
        """getattr"""
        return getattr(self.namespace, attr)

    def __setitem__(self, name, value):
        """setitem"""
        setattr(self.namespace, name, value)

    def __getstate__(self):
        """getstate"""
        return vars(self)

    def __setstate__(self, state):
        """setstate"""
        self.__dict__.update(state)

    def update(self, kwargs):
        """Update parameters"""
        for name, value in kwargs.items():
            setattr(self.namespace, name, value)

        return self


class Environment(object):
    """initialize the enviroment"""
    def __init__(self, args):
        self.args = args
        # init log
        if args.log_path:
            utils.init_log(args.log_path, args.local_rank, args.log_level)
        # init seed
        paddle.seed(args.seed) 
        np.random.seed(args.seed)
        # init place
        if args.use_cuda:
            self.place = "gpu"
        else:
            self.place = "cpu"

        os.environ["FLAGS_paddle_num_threads"] = str(args.threads)
        if not os.path.exists(self.args.model_files):
            os.makedirs(self.args.model_files)
        if not os.path.exists(args.fields_path) or args.preprocess:
            logging.info("Preprocess the data")
            if args.encoding_model in ["ernie-1.0", "ernie-tiny", "ernie-lstm"]:
                tokenizer = ErnieTokenizer.from_pretrained(args.encoding_model)
                self.WORD = ErnieField(
                    "word",
                    pad=tokenizer.pad_token,
                    unk=tokenizer.unk_token,
                    bos=tokenizer.cls_token,
                    eos=tokenizer.sep_token,
                    fix_len=args.fix_len,
                    tokenizer=tokenizer,
                )
                self.WORD.vocab = tokenizer.vocab
                args.feat = None
            else:
                self.WORD = Field(
                    "word",
                    pad=utils.pad,
                    unk=utils.unk,
                    bos=utils.bos,
                    eos=utils.eos,
                    lower=True,
                )
            if args.feat == "char":
                self.FEAT = SubwordField(
                    "chars",
                    pad=utils.pad,
                    unk=utils.unk,
                    bos=utils.bos,
                    eos=utils.eos,
                    fix_len=args.fix_len,
                    tokenize=list,
                )
            elif args.feat == "pos":
                self.FEAT = Field("postag", bos=utils.bos, eos=utils.eos)
            else:
                self.FEAT = None
            self.ARC = Field(
                "head",
                bos=utils.bos,
                eos=utils.eos,
                use_vocab=False,
                fn=utils.numericalize,
            )
            self.REL = Field("deprel", bos=utils.bos, eos=utils.eos)
            if args.feat == "char":
                self.fields = CoNLL(FORM=(self.WORD, self.FEAT), HEAD=self.ARC, DEPREL=self.REL)
            else:
                self.fields = CoNLL(FORM=self.WORD, CPOS=self.FEAT, HEAD=self.ARC, DEPREL=self.REL)

            train = Corpus.load(args.train_data_path, self.fields)

            if not args.encoding_model.startswith("ernie"):
                self.WORD.build(train, args.min_freq)
                self.FEAT.build(train)

            self.REL.build(train)
            if args.local_rank == 0:
                with open(args.fields_path, "wb") as f:
                    logging.info("dumping fileds to disk.")
                    pickle.dump(self.fields, f, protocol=2)
        else:
            logging.info("loading the fields.")
            with open(args.fields_path, "rb") as f:
                self.fields = pickle.load(f)

            if isinstance(self.fields.FORM, tuple):
                self.WORD, self.FEAT = self.fields.FORM
            else:
                self.WORD, self.FEAT = self.fields.FORM, self.fields.CPOS
            self.ARC, self.REL = self.fields.HEAD, self.fields.DEPREL
        
        if args.encoding_model.startswith("ernie"):
            vocab_items = self.WORD.vocab.items()
            args["ernie_vocabs_size"] = len(self.WORD.vocab)
        else:
            vocab_items = self.WORD.vocab.stoi.items()

        self.puncts = np.array([i for s, i in vocab_items if utils.ispunct(s)], dtype=np.int64)

        self.args.update({
            "n_words": len(self.WORD.vocab),
            "n_feats": self.FEAT and len(self.FEAT.vocab),
            "n_rels": len(self.REL.vocab),
            "pad_index": self.WORD.pad_index,
            "unk_index": self.WORD.unk_index,
            "bos_index": self.WORD.bos_index,
            "eos_index": self.WORD.eos_index,
            "feat_pad_index": self.FEAT and self.FEAT.pad_index,
        })
