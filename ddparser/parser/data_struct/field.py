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
本文件定义数据的结构
"""

from collections import Counter

import numpy as np

from ddparser.parser.nets import nn
from ddparser.parser.data_struct import utils
from ddparser.parser.data_struct import Vocab


class RawField(object):
    """Field base class"""
    def __init__(self, name, fn=None):
        super(RawField, self).__init__()

        self.name = name
        self.fn = fn

    def __repr__(self):
        """repr"""
        return "({}): {}()".format(self.name, self.__class__.__name__)

    def preprocess(self, sequence):
        """preprocess"""
        if self.fn is not None:
            sequence = self.fn(sequence)
        return sequence

    def transform(self, sequences):
        """Sequences transform function"""
        return [self.preprocess(seq) for seq in sequences]


class Field(RawField):
    """Field"""
    def __init__(self,
                 name,
                 pad=None,
                 unk=None,
                 bos=None,
                 eos=None,
                 lower=False,
                 use_vocab=True,
                 tokenize=None,
                 tokenizer=None,
                 fn=None):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.tokenizer = tokenizer
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, eos] if token is not None]

    def __repr__(self):
        """repr"""
        s, params = "({}): {}(".format(self.name, self.__class__.__name__), []
        if self.pad is not None:
            params.append("pad={}".format(self.pad))
        if self.unk is not None:
            params.append("unk={}".format(self.unk))
        if self.bos is not None:
            params.append("bos={}".format(self.bos))
        if self.eos is not None:
            params.append("eos={}".format(self.eos))
        if self.lower:
            params.append("lower={}".format(self.lower))
        if not self.use_vocab:
            params.append("use_vocab={}".format(self.use_vocab))
        s += ", ".join(params)
        s += ")"

        return s

    @property
    def pad_index(self):
        """pad index"""
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        """unk index"""
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        """bos index"""
        if self.bos is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        """eos index"""
        if self.eos is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.eos]
        return self.specials.index(self.eos)

    def preprocess(self, sequence):
        """preprocess"""
        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        elif self.tokenizer is not None:
            sequence = self.tokenizer.tokenize(sequence)
            if not sequence: sequence = [self.unk]
        if self.lower:
            sequence = [token.lower() for token in sequence]
        # If the sequence contains special characters, convert it to unk
        sequence = [self.unk if token in self.specials else token for token in sequence]
        return sequence

    def build(self, corpus, min_freq=1):
        """Create vocab based on corpus"""
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(corpus, self.name)
        counter = Counter(token for seq in sequences for token in self.preprocess(seq))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

    def transform(self, sequences):
        """Sequences transform function, such as converting word to id, adding bos tags to sequences, etc."""
        sequences = [self.preprocess(seq) for seq in sequences]
        if self.use_vocab:
            sequences = [self.vocab[seq] for seq in sequences]
        if self.bos:
            sequences = [[self.bos_index] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [self.eos_index] for seq in sequences]

        sequences = [np.array(seq, dtype=np.int64) for seq in sequences]

        return sequences


class SubwordField(Field):
    """SubwordField"""
    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super(SubwordField, self).__init__(*args, **kwargs)

    def build(self, corpus, min_freq=1):
        """Create vocab based on corpus"""
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(corpus, self.name)
        counter = Counter(piece for seq in sequences for token in seq for piece in self.preprocess(token))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

    def transform(self, sequences):
        """Sequences transform function, such as converting word to id, adding bos tags to sequences, etc."""
        sequences = [[self.preprocess(token) for token in seq] for seq in sequences]
        if self.fix_len <= 0:
            self.fix_len = max(len(token) for seq in sequences for token in seq)
        if self.use_vocab:
            sequences = [[[self.vocab[i] for i in token] for token in seq] for seq in sequences]
        if self.bos:
            sequences = [[[self.bos_index]] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [[self.eos_index]] for seq in sequences]
        sequences = [
            nn.pad_sequence([np.array(ids[:self.fix_len], dtype=np.int64) for ids in seq], self.pad_index, self.fix_len)
            for seq in sequences
        ]

        return sequences


class ErnieField(Field):
    """SubwordField"""
    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super(ErnieField, self).__init__(*args, **kwargs)

    def transform(self, sequences):
        """Sequences transform function, such as converting word to id, adding bos tags to sequences, etc."""
        sequences = [[self.preprocess(token) for token in seq] for seq in sequences]
        if self.fix_len <= 0:
            self.fix_len = max(len(token) for seq in sequences for token in seq)
        if self.use_vocab:
            sequences = [[[self.vocab[i] for i in token] for token in seq] for seq in sequences]
        if self.bos:
            sequences = [[[self.bos_index]] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [[self.eos_index]] for seq in sequences]
        sequences = [
            nn.pad_sequence([np.array(ids[:self.fix_len], dtype=np.int64) for ids in seq], self.pad_index, self.fix_len)
            for seq in sequences
        ]

        return sequences
