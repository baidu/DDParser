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
本文件定义dataset, dataloader, datasampler等类及函数
"""

import math
try:
    from collections.abc import Iterable
except:
    from collections import Iterable
from itertools import chain

import numpy as np
from paddle import io

from ddparser.parser.data_struct import utils
from ddparser.parser.nets import nn


class TextDataLoader(object):
    """TextDataLoader"""
    def __init__(self, dataset, batch_sampler, collate_fn, use_data_parallel=False, use_multiprocess=True):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.fields = self.dataset.fields
        self.collate_fn = collate_fn
        self.use_data_parallel = use_data_parallel
        self.dataloader = io.DataLoader.from_generator(capacity=10, return_list=True, use_multiprocess=use_multiprocess)
        self.dataloader.set_batch_generator(self.generator_creator())

    def __call__(self):
        """call"""
        return self.dataloader()

    def generator_creator(self):
        """Returns a generator, each iteration returns a batch of data"""
        def __reader():
            for batch_sample_id in self.batch_sampler:
                batch = []
                raw_batch = self.collate_fn([self.dataset[sample_id] for sample_id in batch_sample_id])
                for data, field in zip(raw_batch, self.fields):
                    if isinstance(data[0], np.ndarray):
                        data = nn.pad_sequence(data, field.pad_index)
                    elif isinstance(data[0], Iterable):
                        data = [nn.pad_sequence(f, field.pad_index) for f in zip(*data)]
                    batch.append(data)
                yield batch

        return __reader

    def __len__(self):
        """Returns the number of batches"""
        return len(self.batch_sampler)


class TextDataset(object):
    """TextDataset"""
    def __init__(self, corpus, fields, n_buckets=None):
        self.corpus = corpus
        self.fields = []
        for field in fields:
            if field is None:
                continue
            if isinstance(field, Iterable):
                self.fields.extend(field)
            else:
                self.fields.append(field)

        for field in self.fields:
            setattr(self, field.name, field.transform(getattr(corpus, field.name)))

        if n_buckets:
            self.lengths = [len(i) + int(bool(field.bos)) for i in corpus]
            self.buckets = dict(zip(*utils.kmeans(self.lengths, n_buckets)))

    def __getitem__(self, index):
        """Returns an iterator containing all fileds of a sample"""
        for field in self.fields:
            yield getattr(self, field.name)[index]

    def __len__(self):
        """The dataset size"""
        return len(self.corpus)

    @classmethod
    def collate_fn(cls, batch):
        """Return batch samples according to field"""
        return (field for field in zip(*batch))


class BucketsSampler(object):
    """BucketsSampler"""
    def __init__(self, buckets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # the number of chunks in each bucket, which is clipped by range [1, len(bucket)]
        self.chunks = []
        for size, bucket in zip(self.sizes, self.buckets):
            max_ch = max(math.ceil(size * len(bucket) / batch_size), 1)
            chunk = min(len(bucket), int(max_ch))
            self.chunks.append(chunk)

    def __iter__(self):
        """Returns an iterator, randomly or sequentially returns a batch id"""
        range_fn = np.random.permutation if self.shuffle else np.arange
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1 for j in range(self.chunks[i])]
            for batch in np.split(range_fn(len(self.buckets[i])), np.cumsum(split_sizes)):
                if len(batch):
                    yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        """Returns the number of batches"""
        return sum(self.chunks)


class SequentialSampler(object):
    """SequentialSampler"""
    def __init__(self, batch_size, corpus_length):
        self.batch_size = batch_size
        self.corpus_length = corpus_length

    def __iter__(self):
        """iter"""
        batch = []
        for i in range(self.corpus_length):
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        else:
            if len(batch):
                yield batch


def batchify(
    dataset,
    batch_size,
    use_data_parallel=False,
    shuffle=False,
    use_multiprocess=True,
    sequential_sampler=False,
):
    """Returns data loader"""
    if sequential_sampler:
        batch_sampler = SequentialSampler(batch_size=batch_size, corpus_length=len(dataset))
    else:
        batch_sampler = BucketsSampler(buckets=dataset.buckets, batch_size=batch_size, shuffle=shuffle)
    loader = TextDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_fn,
        use_data_parallel=use_data_parallel,
        use_multiprocess=use_multiprocess,
    )

    return loader
