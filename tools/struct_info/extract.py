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
'''结构化信息抽取工具'''
import sys
sys.path.insert(0, '../..')

from ddparser import DDParser


class Node:
    """Node class"""
    def __init__(self, id, word, parent, deprel, postag=None):
        self.lefts = []
        self.rights = []
        self.id = int(id)
        self.parent = int(parent) - 1
        self.word = word
        self.deprel = deprel
        self.postag = postag


class Tree(object):
    """Tree Class"""
    def __init__(self, ddp_res):
        self.words = ddp_res['word']
        self.heads = ddp_res['head']
        self.deprels = ddp_res['deprel']
        # self.postags = ddp_res['postag']
        self.build_tree()
        self.ba = ["把", "将"]
        self.bei = ["被"]

    def build_tree(self):
        """Build the tree"""
        self.nodes = []
        for index, (word, head, deprel) in enumerate(zip(self.words, self.heads, self.deprels)):
            node = Node(index, word, head, deprel)
            self.nodes.append(node)
        # set root
        self.root = self.nodes[self.heads.index(0)]
        for node in self.nodes:
            if node.parent == -1:
                continue
            self.add(self.nodes[node.parent], node)

    def add(self, parent: Node, child: Node):
        """Add a child node"""
        if parent.id is None or child.id is None:
            raise IndexError("id is None")
        if parent.id < child.id:
            parent.rights = sorted(parent.rights + [child.id])
        else:
            parent.lefts = sorted(parent.lefts + [child.id])


class FineGrainedInfo(Tree):
    """抽取细粒度结构信息"""
    def __init__(self, ddp_res):
        super(FineGrainedInfo, self).__init__(ddp_res)

    def parse(self):
        """解析句子结构化信息"""
        struct_results = []
        bb_flag = False
        for node in self.nodes:
            bb_structs = self.process_ba(node) + self.process_bei(node)
            bb_flag = len(bb_structs) > 0 or bb_flag
            struct_results += self.process_svo(node, bb_flag)
            struct_results += self.process_pob(node, bb_flag)
            struct_results += self.process_adv(node)
            struct_results += self.process_att(node)
            struct_results += self.process_cmp(node)
            struct_results += self.process_dbl(node)
            struct_results += self.process_vv(node)
            struct_results += self.process_f(node)
            struct_results += self.process_ic(node)
            struct_results += self.process_hed(node)
            struct_results += self.process_dob(node)
            struct_results += bb_structs
        if not struct_results:
            struct_results = self.process_phrase()
        return struct_results

    def process_svo(self, node, flag):
        """获取SVO标签"""
        vs = [node.word]
        ss = []
        os = []
        outputs = []

        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if flag and cnode.deprel == 'POB' and cnode.word in self.ba + self.bei:
                ss = []
                break
            if cnode.deprel == 'DBL':
                return []
            if cnode.deprel == 'SBV':
                ss.append(cnode.word)
                ss += self.process_coo(cnode)
            elif cnode.deprel == 'VOB':
                os.append(cnode.word)
                os += self.process_coo(cnode)
            elif cnode.deprel == 'COO' and cnode.word != node.word:
                vs.append(cnode.word)
            elif cnode.deprel == 'DOB':
                return []

        if len(vs) == 1 and ss and not os and node.deprel == 'ATT' and self.nodes[node.parent].deprel == 'VOB':
            os.append(self.nodes[node.parent].word)

        if ss and os:
            for s in ss:
                for o in os:
                    for v in vs:
                        outputs.append(((s, v, o), 'SVO'))
        else:
            for s in ss:
                for v in vs:
                    outputs.append(((s, v, None), 'SVO'))
            for o in os:
                for v in vs:
                    outputs.append(((None, v, o), 'SVO'))
        return outputs

    def process_att(self, node):
        """处理ATT标签"""
        atts = []
        ns = [node.word]
        outputs = []

        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'ATT':
                atts.append(cnode.word)
                for coo_word in self.process_coo(cnode):
                    atts.append(coo_word)
            elif cnode.deprel == 'COO' and not self.process_att(cnode):
                ns.append(cnode.word)

        for att in atts:
            for n in ns:
                outputs.append(((att, n), 'ATT_N'))

        return outputs

    def process_adv(self, node):
        """处理ADV标签"""
        advs = []
        vs = [node.word]
        outputs = []

        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'ADV' and (not cnode.rights or self.nodes[cnode.rights[0]].deprel != 'POB'):
                advs.append(cnode.word)
                for coo_word in self.process_coo(cnode):
                    advs.append(coo_word)
            elif cnode.deprel == 'COO' and not self.process_adv(cnode):
                vs.append(cnode.word)

        for adv in advs:
            for v in vs:
                outputs.append(((adv, v), 'ADV_V'))

        return outputs

    def process_ba(self, node):
        """处理BA标签"""
        if node.deprel == "POB" and node.word in self.ba and len(node.rights) == 1:
            pnode = self.nodes[node.parent]
            if pnode.rights and self.nodes[pnode.rights[0]].deprel == "VOB":
                for cid in pnode.lefts + pnode.rights:
                    cnode = self.nodes[cid]
                    if cnode.deprel == 'SBV':
                        return [((cnode.word, pnode.word, self.nodes[pnode.rights[0]].word,
                                  self.nodes[node.rights[0]].word), "DOB")]
                return [((None, pnode.word, self.nodes[pnode.rights[0]].word, self.nodes[node.rights[0]].word), "DOB")]
            else:
                for cid in pnode.lefts + pnode.rights:
                    cnode = self.nodes[cid]
                    if cnode.deprel == 'SBV':
                        return [((cnode.word, pnode.word, self.nodes[node.rights[0]].word), "SVO")]
                return [((None, pnode.word, self.nodes[node.rights[0]].word), "SVO")]
        else:
            return []

    def process_bei(self, node):
        """处理BEI标签"""
        outputs = []
        pnode = self.nodes[node.parent]
        if node.deprel == "POB" and node.word in self.bei and pnode.rights and self.nodes[
                pnode.rights[0]].deprel == 'VOB':
            _subject = self.nodes[node.rights[0]].word if node.rights else None
            for cid in pnode.lefts:
                cnode = self.nodes[cid]
                if cnode.deprel == 'SBV':
                    outputs += [((_subject, pnode.word, self.nodes[pnode.rights[0]].word, cnode.word), "DOB")]
            if not outputs:
                outputs += [((_subject, pnode.word, self.nodes[pnode.rights[0]].word, None), "DOB")]

            return outputs
        elif node.deprel == "POB" and node.word in self.bei and len(node.rights) == 1:
            for cid in pnode.lefts:
                cnode = self.nodes[cid]
                if cnode.deprel == 'SBV':
                    outputs += [((self.nodes[node.rights[0]].word, pnode.word, cnode.word), "SVO")]
            if not outputs:
                outputs += [((self.nodes[node.rights[0]].word, pnode.word, None), "SVO")]
            return outputs
        elif node.deprel == "POB" and node.word in self.bei:
            for cid in pnode.lefts:
                cnode = self.nodes[cid]
                if cnode.deprel == 'SBV':
                    outputs += [((None, pnode.word, cnode.word), "SVO")]
            for cid in pnode.rights:
                cnode = self.nodes[cid]
                if cnode.deprel == 'VOB':
                    outputs += [((None, pnode.word, cnode.word), "SVO")]
            return outputs
        else:
            return []

    def process_pob(self, node, bb_flag):
        """处理POB标签"""
        if bb_flag or node.deprel != 'POB':
            return []
        parent = self.nodes[node.parent]
        if parent.parent != -1:
            grandparent = self.nodes[parent.parent]
            return [((node.word, grandparent.word), "ADV_V")]
        else:
            return [((node.word, ), "Phrase")]

    def process_coo(self, node):
        """处理COO标签"""
        outputs = []
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'COO':
                outputs.append(cnode.word)
        return outputs

    def process_phrase(self):
        """获取PHRASE"""
        outputs = []
        outputs.append(((self.nodes[0].word, ), "Phrase"))
        for word in self.process_coo(self.nodes[0]):
            outputs.append(((word, ), "Phrase"))
        return outputs

    def process_cmp(self, node):
        """处理CMP标签"""
        outputs = []
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'CMP':
                outputs.append(((node.word, cnode.word), 'V_CMP'))
        return outputs

    def process_dbl(self, node):
        """处理DBL标签"""
        ss = []
        v = node.word
        o = None
        ds = []
        outputs = []

        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'SBV':
                ss.append(cnode.word)
                ss += self.process_coo(cnode)
            if cnode.deprel == 'DBL':
                if not o:
                    o = cnode.word
                else:
                    ds.append(cnode.word)
                    ds += self.process_coo(cnode)
        if ss and o:
            for s in ss:
                outputs.append(((s, v, o), 'SVO'))
        elif o:
            outputs.append(((None, v, o), 'SVO'))
        if ds:
            for d in ds:
                outputs.append(((o, d, None), 'SVO'))
        return outputs

    def process_vv(self, node):
        """处理VV标签"""
        outputs = []
        sbv_word = None
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'SBV':
                sbv_word = cnode.word
            elif cnode.deprel == 'VV':
                if sbv_word:
                    outputs.append(((sbv_word, cnode.word, None), 'SVO'))
                else:
                    outputs.append(((cnode.word, ), "Phrase"))
        return outputs

    def process_f(self, node):
        """处理F标签"""
        outputs = []
        parent_id = node.parent
        if node.deprel == 'F':
            if parent_id - 1 >= 0 and self.nodes[parent_id - 1].deprel == 'MT' and self.nodes[parent_id -
                                                                                              1].parent == parent_id:
                outputs.append(((self.nodes[parent_id - 1].word, self.nodes[parent_id].word, node.word), "F"))
            else:
                outputs.append(((self.nodes[parent_id].word, node.word), "F"))
        return outputs

    def process_ic(self, node):
        """处理IC标签"""
        outputs = []
        flag = True
        if node.deprel == 'IC':
            for cid in node.lefts + node.rights:
                cnode = self.nodes[cid]
                if cnode.deprel not in ['MT', 'COO', 'IC']:
                    flag = False
                if cnode.deprel in ['COO']:
                    outputs.append(((cnode.word, ), "Phrase"))
            if flag:
                outputs.append(((node.word, ), "Phrase"))
        return outputs

    def process_hed(self, node):
        """处理HED标签"""
        outputs = []
        if node.deprel == 'HED':
            for cid in node.lefts + node.rights:
                cnode = self.nodes[cid]
                if cnode.deprel not in ['MT', 'IC']:
                    return outputs
            outputs.append(((node.word, ), "Phrase"))
        return outputs

    def process_dob(self, node):
        """处理DOB标签"""
        vs = [node.word]
        ss = []
        os = []
        outputs = []
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'SBV':
                ss.append(cnode.word)
                ss += self.process_coo(cnode)
            elif cnode.deprel == 'COO' and cnode.word != node.word:
                vs.append(cnode.word)
            elif cnode.deprel == 'DOB':
                os.append(cnode.word)
        if len(os) != 2:
            return []
        if ss:
            for s in ss:
                for v in vs:
                    outputs.append(((s, v, os[0], os[1]), 'DOB'))
        else:
            for v in vs:
                outputs.append(((None, v, os[0], os[1]), 'DOB'))

        return outputs


class CoarseGrainedInfo(Tree):
    """
    """
    def __init__(self, ddp_res):
        super(CoarseGrainedInfo, self).__init__(ddp_res)

    def parse(self):
        """解析句子结构化信息"""
        struct_results = []
        bb_flag = False
        for node in self.nodes:
            bb_structs = self.process_ba(node) + self.process_bei(node)
            bb_flag = len(bb_structs) > 0 or bb_flag
            struct_results += self.process_svo(node, bb_flag)
            struct_results += self.process_pob(node, bb_flag)
            struct_results += self.process_adv(node)
            struct_results += self.process_att(node)
            struct_results += self.process_cmp(node)
            struct_results += self.process_dbl(node)
            struct_results += self.process_vv(node)
            struct_results += self.process_f(node)
            struct_results += self.process_ic(node)
            struct_results += self.process_hed(node)
            struct_results += self.process_dob(node)
            struct_results += bb_structs
        if not struct_results:
            struct_results = self.process_phrase()
        return struct_results

    def process_svo(self, node, flag):
        """获取SVO标签"""
        vs = [node.word]
        ss = []
        os = []
        outputs = []
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if flag and cnode.deprel == 'POB' and cnode.word in self.ba + self.bei:
                ss = []
                break
            if cnode.deprel == 'DBL':
                return []
            if cnode.deprel == 'SBV':
                ss.append(self.process_sub_term(cnode))
                ss += self.process_coo(cnode)
            elif cnode.deprel == 'VOB':
                os.append(self.process_sub_term(cnode))
                os += self.process_coo(cnode)
            elif cnode.deprel == 'COO' and cnode.word != node.word:
                vs.append(cnode.word)
            elif cnode.deprel == 'DOB':
                return []
        if len(vs) == 1 and ss and not os and node.deprel == 'ATT' and self.nodes[node.parent].deprel == 'VOB':
            os.append(self.nodes[node.parent].word)

        if ss and os:
            for s in ss:
                for o in os:
                    for v in vs:
                        outputs.append(((s, v, o), 'SVO'))
        else:
            for s in ss:
                for v in vs:
                    outputs.append(((s, v, None), 'SVO'))
            for o in os:
                for v in vs:
                    outputs.append(((None, v, o), 'SVO'))
        return outputs

    def process_att(self, node):
        """处理ATT标签"""
        ns = []
        outputs = []
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'ATT' and node.deprel not in ['COO']:
                term = self.process_sub_term(cnode)
                ns.append((term, node.word))
                for coo_word in self.process_coo(cnode):
                    outputs.append(((coo_word, node.word), 'ATT_N'))

        if ns:
            ns += self.process_att_coo(node)
        for n in ns:
            outputs.append((n, 'ATT_N'))

        return outputs

    def process_adv(self, node):
        """处理ADV标签"""
        advs = []
        vs = [node.word]
        outputs = []

        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'ADV' and (not cnode.rights or self.nodes[cnode.rights[0]].deprel != 'POB'):
                advs.append(self.process_sub_term(cnode))
                for coo_word in self.process_coo(cnode):
                    advs.append(coo_word)
            elif cnode.deprel == 'COO' and not self.process_adv(cnode):
                vs.append(cnode.word)

        for adv in advs:
            for v in vs:
                outputs.append(((adv, v), 'ADV_V'))

        return outputs

    def process_ba(self, node):
        """处理BA标签"""
        if node.deprel == "POB" and node.word in self.ba and len(node.rights) == 1:
            pnode = self.nodes[node.parent]
            if pnode.rights and self.nodes[pnode.rights[0]].deprel == "VOB":
                for cid in pnode.lefts + pnode.rights:
                    cnode = self.nodes[cid]
                    if cnode.deprel == 'SBV':
                        return [((self.process_sub_term(cnode), pnode.word,
                                  self.process_sub_term(self.nodes[pnode.rights[0]]),
                                  self.process_sub_term(self.nodes[node.rights[0]])), "DOB")]
                return [((None, pnode.word, self.process_sub_term(self.nodes[pnode.rights[0]]),
                          self.process_sub_term(self.nodes[node.rights[0]])), "DOB")]
            else:
                for cid in pnode.lefts + pnode.rights:
                    cnode = self.nodes[cid]
                    if cnode.deprel == 'SBV':
                        return [((self.process_sub_term(cnode), pnode.word,
                                  self.process_sub_term(self.nodes[node.rights[0]])), "SVO")]
                return [((None, pnode.word, self.process_sub_term(self.nodes[node.rights[0]])), "SVO")]
        else:
            return []

    def process_bei(self, node):
        """处理BEI标签"""
        outputs = []
        pnode = self.nodes[node.parent]
        if node.deprel == "POB" and node.word in self.bei and pnode.rights and self.nodes[
                pnode.rights[0]].deprel == 'VOB':
            _subject = self.process_sub_term(self.nodes[node.rights[0]]) if node.rights else None
            for cid in pnode.lefts:
                cnode = self.nodes[cid]
                if cnode.deprel == 'SBV':
                    outputs += [((_subject, pnode.word, self.process_sub_term(self.nodes[pnode.rights[0]]),
                                  self.process_sub_term(cnode)), "DOB")]
            if not outputs:
                outputs += [((_subject, pnode.word, self.process_sub_term(self.nodes[pnode.rights[0]]), None), "DOB")]
            return outputs
        elif node.deprel == "POB" and node.word in self.bei and len(node.rights) == 1:
            for cid in pnode.lefts:
                cnode = self.nodes[cid]
                if cnode.deprel == 'SBV':
                    outputs += [((self.process_sub_term(self.nodes[node.rights[0]]), pnode.word,
                                  self.process_sub_term(cnode)), "SVO")]
            if not outputs:
                outputs += [((self.process_sub_term(self.nodes[node.rights[0]]), pnode.word, None), "SVO")]
            return outputs
        elif node.deprel == "POB" and node.word in self.bei:
            for cid in pnode.lefts:
                cnode = self.nodes[cid]
                if cnode.deprel == 'SBV':
                    outputs += [((None, pnode.word, self.process_sub_term(cnode)), "SVO")]
            for cid in pnode.rights:
                cnode = self.nodes[cid]
                if cnode.deprel == 'VOB':
                    outputs += [((None, pnode.word, self.process_sub_term(cnode)), "SVO")]
            return outputs
        else:
            return []

    def process_pob(self, node, bb_flag):
        """处理POB标签"""
        if bb_flag or node.deprel != 'POB':
            return []
        parent = self.nodes[node.parent]
        if parent.parent != -1:
            grandparent = self.nodes[parent.parent]
            return [((node.word, grandparent.word), "ADV_V")]
        else:
            return [((node.word, ), "Phrase")]

    def process_coo(self, node):
        """处理COO标签"""
        outputs = []
        term = self.process_sub_term(node)
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'COO':
                cword = self.process_sub_term(cnode)
                if cword == cnode.word:
                    outputs.append(term.replace(node.word, cword))
                else:
                    outputs.append(cword)
        return outputs

    def process_phrase(self):
        """获取PHRASE"""
        outputs = []
        outputs.append(((self.nodes[0].word, ), "Phrase"))
        for word in self.process_coo(self.nodes[0]):
            outputs.append(((word, ), "Phrase"))
        return outputs

    def process_cmp(self, node):
        """处理CMP标签"""
        outputs = []
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'CMP':
                outputs.append(((node.word, cnode.word), 'V_CMP'))
        return outputs

    def process_dbl(self, node):
        """处理DBL标签"""
        ss = []
        v = node.word
        o = None
        ds = []
        outputs = []

        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'SBV':
                ss.append(self.process_sub_term(cnode))
                ss += self.process_coo(cnode)
            if cnode.deprel == 'DBL':
                if not o:
                    o = cnode.word
                else:
                    ds.append(self.process_sub_term(cnode))
                    ds += self.process_coo(cnode)
        if ss and o:
            for s in ss:
                outputs.append(((s, v, o), 'SVO'))
        elif o:
            outputs.append(((None, v, o), 'SVO'))
        if ds:
            for d in ds:
                outputs.append(((o, d, None), 'SVO'))
        return outputs

    def process_vv(self, node):
        """处理VV标签"""
        outputs = []
        sbv_word = None
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'SBV':
                sbv_word = cnode.word
            elif cnode.deprel == 'VV':
                if sbv_word:
                    outputs.append(((sbv_word, cnode.word, None), 'SVO'))
                else:
                    outputs.append(((cnode.word, ), "Phrase"))
        return outputs

    def process_f(self, node):
        """处理F标签"""
        outputs = []
        parent_id = node.parent
        if node.deprel == 'F':
            if parent_id - 1 >= 0 and self.nodes[parent_id - 1].deprel == 'MT' and self.nodes[parent_id -
                                                                                              1].parent == parent_id:
                outputs.append(((self.nodes[parent_id - 1].word, self.nodes[parent_id].word, node.word), "F"))
            else:
                outputs.append(((self.nodes[parent_id].word, node.word), "F"))
        return outputs

    def process_ic(self, node):
        """处理IC标签"""
        outputs = []
        flag = True
        if node.deprel == 'IC':
            for cid in node.lefts + node.rights:
                cnode = self.nodes[cid]
                if cnode.deprel not in ['MT', 'COO', 'IC']:
                    flag = False
                if cnode.deprel in ['COO']:
                    outputs.append(((cnode.word, ), "Phrase"))
            if flag:
                outputs.append(((node.word, ), "Phrase"))
        return outputs

    def process_hed(self, node):
        """处理HED标签"""
        outputs = []
        if node.deprel == 'HED':
            for cid in node.lefts + node.rights:
                cnode = self.nodes[cid]
                if cnode.deprel not in ['MT', 'IC']:
                    return outputs
            outputs.append(((node.word, ), "Phrase"))
        return outputs

    def process_att_coo(self, node):
        """处理ATT标签存在COO的情况"""
        term = []
        n_att = self.process_left_att(node)
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'COO':
                left_coo_att = self.process_left_att(cnode)
                if not left_coo_att:
                    term.append((n_att, cnode.word))
                else:
                    term.append((left_coo_att, cnode.word))
        return term

    def process_left_att(self, node):
        """获取节点的左侧ATT字段"""
        left_att = ""
        for cid in node.lefts:
            cnode = self.nodes[cid]
            if cnode.deprel == 'ATT':
                left_att += self.process_sub_term(cnode)

        return left_att

    def process_sub_term(self, node):
        """获取节点及其全部子节点组成的片段"""
        sub_tokens = self.inorder_traversal(node)
        if sub_tokens[0][1] == 'MT' and len(sub_tokens) > 1:
            sub_tokens.pop(0)
        if sub_tokens[-1][1] == 'MT' and len(sub_tokens) > 1:
            sub_tokens.pop(-1)
        tokens = []
        if len(sub_tokens) == 1:
            return sub_tokens[0][0]
        tokens, _ = zip(*sub_tokens)

        return "".join(tokens)

    def inorder_traversal(self, node):
        """中序遍历"""
        lf_list = []
        rf_list = []
        for ln in node.lefts:
            if self.nodes[ln].deprel not in ['COO']:
                lf_list += self.inorder_traversal(self.nodes[ln])
        for rn in node.rights:
            if self.nodes[rn].deprel not in ['COO']:
                rf_list += self.inorder_traversal(self.nodes[rn])

        return lf_list + [(node.word, node.deprel)] + rf_list

    def process_dob(self, node):
        """获取DOB标签"""
        vs = [node.word]
        ss = []
        os = []
        outputs = []
        for cid in node.lefts + node.rights:
            cnode = self.nodes[cid]
            if cnode.deprel == 'SBV':
                ss.append(self.process_sub_term(cnode))
                ss += self.process_coo(cnode)
            elif cnode.deprel == 'COO' and cnode.word != node.word:
                vs.append(cnode.word)
            elif cnode.deprel == 'DOB':
                os.append(self.process_sub_term(cnode))
        if len(os) != 2:
            return []
        if ss:
            for s in ss:
                for v in vs:
                    outputs.append(((s, v, os[0], os[1]), 'DOB'))
        else:
            for v in vs:
                outputs.append(((None, v, os[0], os[1]), 'DOB'))
        return outputs


if __name__ == "__main__":
    ddp = DDParser(encoding_model='transformer')
    text = ["百度是一家高科技公司"]
    ddp_res = ddp.parse(text)
    print(ddp_res)
    # 细粒度
    fine_info = FineGrainedInfo(ddp_res[0])
    print("细粒度：", fine_info.parse())
    # 粗粒度
    coarse_info = CoarseGrainedInfo(ddp_res[0])
    print("粗粒度：", coarse_info.parse())
