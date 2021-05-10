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
"""parser"""

from .model import decode
from .model import epoch_evaluate
from .model import epoch_predict
from .model import epoch_train
from .model import load
from .model import loss_function
from .model import save
from .model import Model
from .config import ArgConfig
from .config import Environment

__all__ = [
    'decode', 'epoch_evaluate', 'epoch_predict', 'epoch_train', 'load', 'loss_function', 'save', 'Model', 'ArgConfig',
    'Environment'
]
