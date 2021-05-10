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
本文件定义指标的相关类
"""

import numpy as np
from paddle.fluid import layers

from ddparser.parser.nets import nn


class Metric(object):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()

        self.eps = eps
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        """repr"""
        return "UAS: {:6.2%} LAS: {:6.2%}".format(self.uas, self.las)

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        """call"""
        arc_mask = nn.masked_select(arc_preds == arc_golds, mask)
        rel_mask = layers.logical_and(nn.masked_select(rel_preds == rel_golds, mask), arc_mask)
        self.total += len(arc_mask)
        self.correct_arcs += np.sum(arc_mask.numpy()).item()
        self.correct_rels += np.sum(rel_mask.numpy()).item()

    @property
    def score(self):
        """score"""
        return self.las

    @property
    def uas(self):
        """uas"""
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        """las"""
        return self.correct_rels / (self.total + self.eps)

    def __lt__(self, other):
        """lt"""
        return self.score < other

    def __le__(self, other):
        """le"""
        return self.score <= other

    def __ge__(self, other):
        """ge"""
        return self.score >= other

    def __gt__(self, other):
        """gt"""
        return self.score > other