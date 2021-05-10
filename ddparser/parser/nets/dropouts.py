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
"""本文件定义dropout的相关类"""

import numpy as np
from paddle.fluid import layers
from paddle.fluid import dygraph


class SharedDropout(dygraph.Layer):
    """SharedDropout"""
    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        """Forward network"""
        if self.training and self.p > 0:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= layers.unsqueeze(mask, axes=[1]) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        """Generate the mask matrix of the dropout by the input."""
        mask = layers.uniform_random(shape=x.shape, min=0, max=1) >= p
        mask = layers.cast(mask, 'float32')
        mask = mask / (1 - p)
        return mask


class IndependentDropout(dygraph.Layer):
    """IndependentDropout"""
    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def forward(self, *items):
        """Forward network"""
        if self.training and self.p > 0:
            masks = [layers.uniform_random(shape=x.shape[:2], min=0, max=1) >= self.p for x in items]
            masks = [layers.cast(x, 'float32') for x in masks]
            total = layers.elementwise_add(*masks)
            scale = len(items) / layers.elementwise_max(total, layers.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * layers.unsqueeze(mask, axes=[-1]) for item, mask in zip(items, masks)]
        return items
