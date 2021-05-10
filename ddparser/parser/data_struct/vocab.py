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
本文件定义Vocab类
"""

from collections import defaultdict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import six


class Vocab(object):
    """Vocab"""
    def __init__(self, counter, min_freq=1, specials=None, unk_index=0):
        self.itos = list(specials) if specials else []
        self.stoi = defaultdict(lambda: unk_index)
        self.stoi.update({token: i for i, token in enumerate(self.itos)})
        self.extend([token for token, freq in counter.items() if freq >= min_freq])
        self.unk_index = unk_index
        self.n_init = len(self)

    def __len__(self):
        """Returns the size of the vocabulary"""
        return len(self.itos)

    def __getitem__(self, key):
        """According to the key or index, return the index and key"""
        if isinstance(key, six.string_types):
            return self.stoi[key]
        elif not isinstance(key, Iterable):
            return self.itos[key]
        elif isinstance(key[0], six.string_types):
            return [self.stoi[i] for i in key]
        else:
            return [self.itos[i] for i in key]

    def __contains__(self, token):
        """contains"""
        return token in self.stoi

    def __getstate__(self):
        """getstate"""
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        """setstate"""
        stoi = defaultdict(lambda: self.unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def extend(self, tokens):
        """Update tokens to itos and stoi"""
        self.itos.extend(sorted(set(tokens).difference(self.stoi)))
        self.stoi.update({token: i for i, token in enumerate(self.itos)})
