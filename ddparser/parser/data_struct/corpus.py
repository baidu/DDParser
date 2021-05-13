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
本文件定义数据集相关的类和对象
"""

from collections import namedtuple
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
    from io import open

from ddparser.parser.data_struct import Field

CoNLL = namedtuple(typename='CoNLL',
                   field_names=['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'])
CoNLL.__new__.__defaults__ = tuple([None] * 10)


class Sentence(object):
    """Sentence"""
    def __init__(self, fields, values):
        for field, value in zip(fields, values):
            if isinstance(field, Iterable):
                for j in range(len(field)):
                    setattr(self, field[j].name, value)
            else:
                setattr(self, field.name, value)
        self.fields = fields

    @property
    def values(self):
        """Returns an iterator containing all the features of one sentence"""
        for field in self.fields:
            if isinstance(field, Iterable):
                yield getattr(self, field[0].name)
            else:
                yield getattr(self, field.name)

    def __len__(self):
        """Get sentence length"""
        return len(next(iter(self.values)))

    def __repr__(self):
        """repr"""
        return '\n'.join('\t'.join(map(str, line)) for line in zip(*self.values)) + '\n'

    def get_result(self):
        """Returns json style result"""
        output = {}
        for field in self.fields:
            if isinstance(field, Iterable) and not field[0].name.isdigit():
                output[field[0].name] = getattr(self, field[0].name)
            elif not field.name.isdigit():
                output[field.name] = getattr(self, field.name)
        return output


class Corpus(object):
    """Corpus"""
    def __init__(self, fields, sentences):
        super(Corpus, self).__init__()

        self.fields = fields
        self.sentences = sentences

    def __len__(self):
        """Returns the data set size"""
        return len(self.sentences)

    def __repr__(self):
        """repr"""
        return '\n'.join(str(sentence) for sentence in self)

    def __getitem__(self, index):
        """Get the sentence according to the index"""
        return self.sentences[index]

    def __getattr__(self, name):
        """Get the value of name and return an iterator"""
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for sentence in self.sentences:
            yield getattr(sentence, name)

    def __setattr__(self, name, value):
        """Add a property"""
        if name in ['fields', 'sentences']:
            self.__dict__[name] = value
        else:
            for i, sentence in enumerate(self.sentences):
                setattr(sentence, name, value[i])

    @classmethod
    def load(cls, path, fields):
        """Load data from path to generate corpus"""
        start, sentences = 0, []
        fields = [fd if fd is not None else Field(str(i)) for i, fd in enumerate(fields)]
        with open(path, 'r', encoding='utf-8') as f:
            lines = [
                line.strip() for line in f.readlines()
                if not line.startswith('#') and (len(line) == 1 or line.split()[0].isdigit())
            ]
        for i, line in enumerate(lines):
            if not line:
                values = list(zip(*[j.split('\t') for j in lines[start:i]]))
                if values:
                    sentences.append(Sentence(fields, values))
                start = i + 1

        return cls(fields, sentences)

    @classmethod
    def load_lac_results(cls, inputs, fields):
        """Load data from lac results to generate corpus"""
        sentences = []
        fields = [fd if fd is not None else Field(str(i)) for i, fd in enumerate(fields)]
        for _input in inputs:
            if isinstance(_input[0], list):
                tokens, poss = _input
            else:
                tokens = _input
                poss = ['-'] * len(tokens)
            values = [list(range(1,
                                 len(tokens) + 1)), tokens, tokens, poss, poss] + [['-'] * len(tokens)
                                                                                   for _ in range(5)]

            sentences.append(Sentence(fields, values))
        return cls(fields, sentences)

    @classmethod
    def load_word_segments(cls, inputs, fields):
        """Load data from word segmentation results to generate corpus"""
        fields = [fd if fd is not None else Field(str(i)) for i, fd in enumerate(fields)]
        sentences = []
        for tokens in inputs:
            values = [list(range(1, len(tokens) + 1)), tokens, tokens] + [['-'] * len(tokens) for _ in range(7)]

            sentences.append(Sentence(fields, values))
        return cls(fields, sentences)

    def save(self, path):
        """Dumping corpus to disk"""
        with open(path, 'w') as f:
            f.write(u"{}\n".format(self))

    def _print(self):
        """Print self"""
        print(self)

    def get_result(self):
        """Get result"""
        output = []
        for sentence in self:
            output.append(sentence.get_result())
        return output
