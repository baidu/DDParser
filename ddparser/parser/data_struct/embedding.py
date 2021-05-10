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
本文件定义Embedding类，用于文件中加载预训练词向量
"""

import numpy as np


class Embedding(object):
    """Embedding"""
    def __init__(self, tokens, vectors, unk=None):
        super(Embedding, self).__init__()

        self.tokens = tokens
        self.vectors = np.array(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

    def __len__(self):
        """Vocabulary size"""
        return len(self.tokens)

    def __contains__(self, token):
        """Is token in pretrained"""
        return token in self.pretrained

    @property
    def dim(self):
        """Word vector size"""
        return self.vectors.shape[1]

    @property
    def unk_index(self):
        """unk index"""
        if self.unk is not None:
            return self.tokens.index(self.unk)
        else:
            raise AttributeError

    @classmethod
    def load(cls, path, unk=None):
        """Load embedding and token from path"""
        with open(path, 'r') as f:
            lines = [line for line in f.readlines()]
        splits = [line.split() for line in lines]
        tokens, vectors = zip(*[(s[0], list(map(float, s[1:]))) for s in splits])

        return cls(tokens, vectors, unk=unk)
