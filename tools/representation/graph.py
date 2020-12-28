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
"""本文件定义将依存分析的弧转化为邻接矩阵的函数"""

import re
from itertools import chain

import numpy as np


def get_arcs_and_head(ddp_result, tokens):
    """
    将ddparser输出的依存弧映射到用户自定义切词结果上，返回映射后的依存弧和核心词索引。

    Arags：
        ddp_result: ddparser结果
        tokens: 用户自定义切词
    Returns:
        arc_tails: arc_tails表示映射后所有弧尾的索引
        arc_heads: arc_heads表示映射后所有弧头的索引
        head_id: 表示映射后核心词的所有
    """
    words = ddp_result['word']
    heads = ddp_result['head']
    # 为了运算方便，父节点索引减一
    heads = [i - 1 for i in heads]
    mapping = _get_mapping(words, tokens)
    arc_tails, arc_heads = _get_arcs(mapping, heads)
    head_id = _get_head_id(ddp_result, mapping)
    return (arc_tails, arc_heads), head_id


def get_arcs_and_head_in_wordpiece(ddp_result, tokens):
    """
    当用户的切词使用的是wordpiece时，将ddparser输出的依存弧映射到用户自定义切词结果上，返回映射后的依存弧和核心词索引。

    Arags：
        ddp_result: ddparser结果
        tokens: 用户自定义切词
    Returns:
        arc_tails: arc_tails表示映射后所有弧尾的索引
        arc_heads: arc_heads表示映射后所有弧头的索引
        head_id: 表示映射后核心词的所有
    """
    words = [s.lower() for s in ddp_result['word']]
    heads = ddp_result['head']
    # 为了运算方便，父节点索引减一
    heads = [i - 1 for i in heads]
    merge_idxs, merge_tokens = _merge_wordpiece_tokens(tokens)
    # replace [UNK]
    words, merge_tokens = _replace_unk(words, merge_tokens)
    assert "".join(words) == "".join(merge_tokens)
    mapping = _get_mapping(words, merge_tokens)
    re_index = list(range(len(tokens)))
    for n, i in enumerate(merge_idxs):
        re_index[n] = mapping[i]
    arc_tails, arc_heads = _get_arcs(re_index, heads)
    head_id = _get_head_id(ddp_result, re_index)
    return (arc_tails, arc_heads), head_id


def get_adj_of_one_sent(arcs, length, max_len=None):
    """
    将弧转化为邻接矩阵

    Arags：
        arcs: 弧
        length: token数量
        max_len: 最大token的数量
    Returns:
        graph: 转化后的邻接矩阵
    """
    if max_len is None:
        max_len = length
    arc_tails, arc_heads = [], []
    for arc_tail, arc_head in zip(*arcs):
        if arc_tail < max_len and arc_head < max_len:
            arc_tails.append(arc_tail)
            arc_heads.append(arc_head)
    if not arc_tails:
        arc_tails.append(0)
        arc_heads.append(0)

    graph = np.zeros((max_len, max_len), dtype="int32")
    for arc_tail, arc_head in zip(arc_tails, arc_heads):
        graph[arc_tail, arc_head] = 1
    for i in range(max_len):
        graph[i, i] = 1
    return graph


def get_adj_of_one_sent_in_ernie(arcs, length, max_len=None):
    """
    当用户模型是ernie时，将弧转化为邻接矩阵(自动补齐[CLS],[SEP])

    Arags：
        arcs: 弧
        length: token数量
        max_len: 最大token的数量
    Returns:
        graph: 转化后的邻接矩阵
    """
    if max_len is None:
        max_len = length
    arc_tails, arc_heads = [], []
    for arc_tail, arc_head in zip(*arcs):
        if arc_tail < max_len and arc_head < max_len:
            arc_tails.append(arc_tail + 1)
            arc_heads.append(arc_head + 1)
    if not arc_tails:
        arc_tails.append(0)
        arc_heads.append(0)

    graph = np.zeros((max_len + 2, max_len + 2), dtype="int32")
    for arc_tail, arc_head in zip(arc_tails, arc_heads):
        graph[arc_tail, arc_head] = 1
    for i in range(max_len + 2):
        if i not in [0, max_len + 1]:
            graph[i, i] = 1
    return graph


def get_adj_of_two_sent_in_ernie(arcs_a, length_a, arcs_b, length_b, max_len=None):
    """
    当用户模型是ernie且输入为两条句子拼接在一起时，将弧转化为邻接矩阵(自动补齐[CLS],[SEP])

    Arags：
        arcs_a: 句子a的弧
        length_a: 句子a的token数量
        arcs_b: 句子b的弧
        length_b: 句子b的token数量
        max_len: 最大token的数量
    Returns:
        graph: 转化后的邻接矩阵
    """
    if max_len is None:
        max_len_a = length_a
    else:
        max_len_a = max_len
    arc_tails, arc_heads = [], []
    for arc_tail, arc_head in zip(*arcs_a):
        if arc_tail < max_len_a and arc_head < max_len_a:
            arc_tails.append(arc_tail + 1)
            arc_heads.append(arc_head + 1)

    if max_len is None:
        max_len_b = length_b
    else:
        max_len_b = max_len
    for arc_tail, arc_head in zip(*arcs_b):
        if arc_tail < max_len_b and arc_head < max_len_b:
            arc_tails.append(arc_tail + max_len_a + 2)
            arc_heads.append(arc_head + max_len_a + 2)
    if not arc_tails:
        arc_tails.append(0)
        arc_heads.append(0)

    graph = np.zeros((max_len_a + max_len_b + 3, max_len_a + max_len_b + 3), dtype="int32")
    for arc_tail, arc_head in zip(arc_tails, arc_heads):
        graph[arc_tail, arc_head] = 1
    for i in range(max_len_a + max_len_b + 3):
        if i not in [0, max_len_a + 1, max_len_a + max_len_b + 2]:
            graph[i, i] = 1
    return graph


def pad_batch_graphs(graphs, max_len=None):
    """
    padding batch graphs

    Arags：
        graphs: 未填充的邻接矩阵
        max_len: 最大长度
    Returns:
        out_tensor: 填充后的邻接矩阵
    """
    if max_len is None:
        max_len = max([s.shape[0] for s in graphs])
    out_dims = (len(graphs), max_len, max_len)
    out_tensor = np.full(out_dims, 0, dtype=np.int64)
    for i, tensor in enumerate(graphs):
        length = min(tensor.shape[0], max_len)
        out_tensor[i, :length, :length] = tensor
    return out_tensor


def _get_arcs(mapping, heads):
    """
    映射函数，获取映射后的弧
    """
    arc_tails, arc_heads = [], []
    for n, i in enumerate(mapping):
        if i == -1 or heads[i] == -1:
            continue
        for m, j in enumerate(mapping):
            if j != -1 and j == heads[i]:
                arc_tails.append(n)
                arc_heads.append(m)
                arc_tails.append(m)
                arc_heads.append(n)
    return arc_tails, arc_heads


def _get_mapping(words, tokens):
    """
    获取映射数组
    """
    index = list(range(len(tokens)))
    tmp_ddp = words[0]
    tmp_tok = tokens[0]
    ddp_idx = 0
    tok_idx = 0
    while ddp_idx < len(words) - 1 or tok_idx < len(tokens) - 1:
        if tmp_ddp == tmp_tok:
            index[tok_idx] = ddp_idx
            tok_idx += 1
            ddp_idx += 1
            tmp_ddp += words[ddp_idx]
            tmp_tok += tokens[tok_idx]
        elif tmp_ddp > tmp_tok:
            # index[tok_idx] = ddp_idx
            index[tok_idx] = -1
            tok_idx += 1
            tmp_tok += tokens[tok_idx]
        elif tmp_ddp < tmp_tok:
            # index[tok_idx] = ddp_idx
            index[tok_idx] = -1
            ddp_idx += 1
            tmp_ddp += words[ddp_idx]
    else:
        index[tok_idx] = ddp_idx
    return index


def _merge_wordpiece_tokens(tokens):
    """合并被wordpiece切散的token"""
    assert len(tokens) >= 1
    idxs = []
    m_tokens = []
    cur_token = ""
    for token in tokens:
        if cur_token == "":
            cur_token += token
            idxs.append(0)
            continue
        if token.startswith("##"):
            cur_token += token[2:]
            idxs.append(idxs[-1])
        else:
            m_tokens.append(cur_token)
            cur_token = token
            idxs.append(idxs[-1] + 1)
    else:
        m_tokens.append(cur_token)
    return idxs, m_tokens


def _get_head_id(ddp_result, mapping):
    """获取映射后核心词索引"""
    heads = ddp_result['head']
    try:
        head_id = mapping.index(heads.index(0))
    except:
        head_id = len(mapping) - 1
    return head_id


def _replace_unk(words, tokens):
    """将[UNK]符号替换为原始符号"""
    if '[UNK]' not in tokens:
        return words, tokens
    words = [_replace_escape(word) for word in words]
    query = "".join(words)
    new_tokens = []
    for token in tokens:
        if token != '[UNK]':
            new_tokens.append(_replace_escape(token))
        else:
            new_tokens.append('(.+?)')
    matchs = re.match("".join(new_tokens) + "$", query)
    if matchs is None:
        raise "unkonwn error"

    for match in matchs.groups():
        new_tokens[new_tokens.index('(.+?)')] = match
    return words, new_tokens


ESCAPE_DICT = {
    '(': '（',
    ')': '）',
    '[': '【',
    ']': '】',
    '+': '＋',
    '?': '？',
    '*': '×',
    '{': '｛',
    '}': '｝',
    '.': '．',
}


def _replace_escape(string):
    """将正则中的转义字符替换为全角字符"""
    for k, v in ESCAPE_DICT.items():
        string = string.replace(k, v)
    return string


def transfor_head_id_for_ernie(head_id_a, length_a, head_id_b=None, length_b=None):
    """
    当用户模型为ernie时, 获取新的核心词位置（由于拼接[CLS], [SEP]）
    """
    if head_id_b is None or length_b is None:
        return min(head_id_a + 1, length_a)
    else:
        return (min(head_id_a + 1, length_a), min(length_a + head_id_b + 2, length_a + length_b + 1))


if __name__ == "__main__":
    d = {'word': ['10086', '话费', '清单', '查询'], 'head': [2, 3, 4, 0], 'deprel': ['ATT', 'ATT', 'VOB', 'HED']}

    t = ['1008', '##6', '话', '费', '清', '单', '查', '询']
    print(_merge_wordpiece_tokens(t))