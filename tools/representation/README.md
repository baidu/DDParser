## 基于依存分析树和GAT的句子表示工具
最近几年图神经网络(GNN)越来越多被应用到NLP任务，而DDParser可以输出句子的依存句法分析树，由于树是图的一种特例，那么很自然的可以将GNN基于依存句法分析结果应用到句子表示上。<br>
### 实现原理
本工具提出一种可以快速将依存句法分析特征应用到下游网络的方法。
* 第一步，利用DDParser对用户原始数据处理，得到依存句法分析结果。
* 第二步，修改原始模型的Dataloader，利用本工具提供的接口得到文本的邻接矩阵<img src="https://latex.codecogs.com/svg.latex?\mathrm{A}">和核心词索引<img src="https://latex.codecogs.com/svg.latex?\mathrm{h}">。
* 第三步，在模型得到句子的表示<img src="https://latex.codecogs.com/svg.latex?\mathrm{X}">后，修改模型结构引入图注意力网络([Graph Attention Networks, GAT](https://arxiv.org/abs/1710.10903))，将<img src="https://latex.codecogs.com/svg.latex?\mathrm{A}">和<img src="https://latex.codecogs.com/svg.latex?\mathrm{X}">作为GAT网络输入得到包含句子结构信息的句子表示<img src="https://latex.codecogs.com/svg.latex?\bar{X}">。 
* 第四步，对<img src="https://latex.codecogs.com/svg.latex?\bar{X}">应用可分为两种方式。
  - 1.将<img src="https://latex.codecogs.com/svg.latex?\bar{X}">和<img src="https://latex.codecogs.com/svg.latex?\mathrm{X}">拼接后应用到下游网络。
  - 2.通过<img src="https://latex.codecogs.com/svg.latex?\mathrm{h}">获取句子核心词表示，将核心词的表示应用到下游网络。

经过实验，发现本工具可以有效提升相似度、事件抽取等任务效果。

### 示例
为了方便用户快速使用本工具，我们基于ERNIE1.0模型和LCQMC数据集提供了一个示例。
用户需要自行下载[ERNIE1.0](https://github.com/PaddlePaddle/ERNIE)模型和[LCQMC](http://icrc.hitsz.edu.cn/info/1037/1146.htm)数据集，并通过`preprocess_data.py`对数据预处理。<br>
```python
cd demo/LCQMC
python preprocess_data.py
```
在准备好模型和数据后，用户需要修改`demo/run_demo.sh`中`MODEL_PATH`和`TASK_DATA_PATH`字段指明模型和数据的路径，通过下列命令运行示例。
```python
sh run_demo.sh
```
我们与基线模型做了效果对比，结果如下：

| 模型                    | dev acc | test acc |
| :---------------------- | :------ | :------- |
| ERNIE 1.0 Base          | 90.1    | 87.4     |
| ERNIE 1.0 Base + 本工具 | 90.3    | 87.6     |

注：每组实验运行5次，取每次实验中最好的dev acc及对应test acc的均值作为对比结果。

### 进阶使用
#### 目录结构
```text
.
├── README.md
├── gnn.py             # GAT网络代码
├── graph.py           # 数据处理，邻接矩阵生成相关代码
├── utils.py           # 工具类
├── demo               # 示例项目
```
#### 快速使用
以demo项目为例，我们将介绍如何应用本工具到用户的项目上。<br>
用户在预处理数据后，需要修改`demo/ERNIE/reader/task_reader.py`文件中的`ClassifyReader._read_tsv`函数，使用`pandas.read_csv`函数代替`csv.reader`函数读取数据。
```python
def _read_tsv(self, input_file, quotechar=None):
    """Reads a tab separated value file."""
    reader = pd.read_csv(input_file, sep='\t')
    headers = reader.columns
    text_indices = [
        index for index, h in enumerate(headers) if h != "label"
    ]
    Example = namedtuple('Example', headers)
    examples = []
    for _, line in reader.iterrows():
        for index, text in enumerate(line):
            if index in text_indices:
                if self.for_cn:
                    line[index] = text.replace(' ', '')
                else:
                    line[index] = text
        example = Example(*line.tolist())
        examples.append(example)
    return examples
```
其次，修改`BaseReader._convert_example_to_record`函数，在原有特征上增加邻接矩阵和核心词索引。
```python
def _convert_example_to_record(self, example, max_seq_length, tokenizer):
    """Converts a single `Example` into a single `Record`."""

    text_a = tokenization.convert_to_unicode(example.text_a)
    ddp_res_a = eval(example.ddp_res_a)
    tokens_a = tokenizer.tokenize(text_a)
    # 获取句子a的弧及核心词索引
    arcs_a, head_id_a = get_arcs_and_head_in_wordpiece(ddp_res_a, tokens_a)
    tokens_b = None

    has_text_b = False
    if isinstance(example, dict):
        has_text_b = "text_b" in example.keys()
    else:
        has_text_b = "text_b" in example._fields

    if has_text_b:
        text_b = tokenization.convert_to_unicode(example.text_b)
        ddp_res_b = eval(example.ddp_res_b)
        tokens_b = tokenizer.tokenize(text_b)
        # 获取句子b的弧及核心词索引
        arcs_b, head_id_b = get_arcs_and_head_in_wordpiece(
            ddp_res_b, tokens_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # 获取句子a和b组成的邻接矩阵
        adjacency_matrix = get_adj_of_two_sent_in_ernie(
            arcs_a, len(tokens_a), arcs_b, len(tokens_b))
        # 获取映射后核心词索引
        head_ids = transfor_head_id_for_ernie(head_id_a, len(tokens_a),
                                              head_id_b, len(tokens_b))
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
        adjacency_matrix = get_adj_of_one_sent_in_ernie(
            arcs_a, len(tokens_a))
        head_ids = transfor_head_id_for_ernie(head_id_a, len(tokens_a))

    # 以下代码省略
    .....
    .....
    # 增加adjacency_matrix和head_ids特征
    if self.is_inference:
        Record = namedtuple('Record', [
            'token_ids', 'text_type_ids', 'position_ids',
            'adjacency_matrix', 'head_ids'
        ])
        record = Record(token_ids=token_ids,
                        text_type_ids=text_type_ids,
                        position_ids=position_ids,
                        adjacency_matrix=adjacency_matrix,
                        head_ids=head_ids)
    else:
        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple('Record', [
            'token_ids', 'text_type_ids', 'position_ids', 'label_id', 'qid',
            'adjacency_matrix', 'head_ids'
        ])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(token_ids=token_ids,
                        text_type_ids=text_type_ids,
                        position_ids=position_ids,
                        label_id=label_id,
                        qid=qid,
                        adjacency_matrix=adjacency_matrix,
                        head_ids=head_ids)
    return record
```
修改`demo/ERNIE/batching.py中pad_batch_data`函数，增加`max_len`参数。
```python
def pad_batch_data(insts,
                   max_len=None,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    if max_len is None:
        max_len = max(len(inst) for inst in insts)
    # 以下代码省略
    ......
```
在`demo/ERNIE/reader/task_reader.py`的`ClasssifyReader._pad_batch_records`中调用`pad_batch_data`函数时增加`max_len=self.max_seq_len`参数(为了将所有batch的长度都填充到最大长度，方便后面GAT网络计算)。
```python
def _pad_batch_records(self, batch_records):
    batch_token_ids = [record.token_ids for record in batch_records]
    batch_text_type_ids = [record.text_type_ids for record in batch_records]
    batch_position_ids = [record.position_ids for record in batch_records]
    # 增加batch_adjacency_matrix
    batch_adjacency_matrix = [
        record.adjacency_matrix for record in batch_records
    ]
    # 增加batch_head_ids
    batch_head_ids = np.array([record.head_ids
                                for record in batch_records]).astype("int64")

    if not self.is_inference:
        batch_labels = [record.label_id for record in batch_records]
        if self.is_classify:
            batch_labels = np.array(batch_labels).astype("int64").reshape(
                [-1, 1])
        elif self.is_regression:
            batch_labels = np.array(batch_labels).astype("float32").reshape(
                [-1, 1])

        if batch_records[0].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape(
                [-1, 1])
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])

    # padding
    # 增加max_len=self.max_seq_len，将所有batch的长度都填充到最大长度
    padded_token_ids, input_mask = pad_batch_data(batch_token_ids,
                                                  max_len=self.max_seq_len,
                                                  pad_idx=self.pad_id,
                                                  return_input_mask=True)
    padded_text_type_ids = pad_batch_data(batch_text_type_ids,
                                          max_len=self.max_seq_len,
                                          pad_idx=self.pad_id)
    padded_position_ids = pad_batch_data(batch_position_ids,
                                          max_len=self.max_seq_len,
                                          pad_idx=self.pad_id)
    padded_task_ids = np.ones_like(padded_token_ids,
                                    dtype="int64") * self.task_id
    padded_adjacency_matrix = pad_batch_graphs(batch_adjacency_matrix,
                                                max_len=self.max_seq_len)

    return_list = [
        padded_token_ids,
        padded_text_type_ids,
        padded_position_ids,
        padded_task_ids,
        input_mask,
    ]
    if not self.is_inference:
        return_list += [batch_labels, batch_qids]
    # 增加padded_adjacency_matrix和batch_head_ids的返回
    return_list += [padded_adjacency_matrix, batch_head_ids]

    return return_list
```
最后，修改`ERNIE/finetune/classifier.py`中的`create_model`函数，增加GAT网络。
```python
def create_model(args,
                 pyreader_name,
                 ernie_config,
                 is_prediction=False,
                 task_name="",
                 is_classify=False,
                 is_regression=False,
                 ernie_version="1.0"):
    if is_classify:
        # 增加邻接矩阵和核心词的shape
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1], [-1, 1], [-1, 1],
                    [-1, args.max_seq_len, args.max_seq_len], [-1, 2]],
            dtypes=[
                'int64', 'int64', 'int64', 'int64', 'float32', 'int64',
                'int64', 'int64', 'int64'
            ],
            lod_levels=[0, 0, 0, 0, 0, 0, 0, 0, 0],
            name=task_name + "_" + pyreader_name,
            use_double_buffer=True)
    elif is_regression:
        pyreader = fluid.layers.py_reader(capacity=50,
                                          shapes=[[-1, args.max_seq_len, 1],
                                                  [-1, args.max_seq_len, 1],
                                                  [-1, args.max_seq_len, 1],
                                                  [-1, args.max_seq_len, 1],
                                                  [-1, args.max_seq_len, 1],
                                                  [-1, 1], [-1, 1]],
                                          dtypes=[
                                              'int64', 'int64', 'int64',
                                              'int64', 'float32', 'float32',
                                              'int64'
                                          ],
                                          lod_levels=[0, 0, 0, 0, 0, 0, 0],
                                          name=task_name + "_" + pyreader_name,
                                          use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, task_ids, input_mask, labels, qids, adj_mat,
     head_ids) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(src_ids=src_ids,
                       position_ids=pos_ids,
                       sentence_ids=sent_ids,
                       task_ids=task_ids,
                       input_mask=input_mask,
                       config=ernie_config,
                       use_fp16=args.use_fp16)
    erinie_output = ernie.get_sequence_output()
    cls_feats = ernie.get_pooled_output()

    # 增加GAT网络
    gat = gnn.GAT(768, 100, 200, 0.0, 0.1, 12, 2)
    # 将ernie的表示和邻接矩阵输入到gat网络中得到包含句子结构信息的表示
    gat_emb = gat.forward(erinie_output, adj_mat)
    # 提取核心词的表示
    gat_emb = utils.index_sample(gat_emb, head_ids)
    # 将[CLS]和核心词的表示拼接，供下游网络使用
    cls_feats = fluid.layers.concat([cls_feats, gat_emb], axis=1)

    #以下代码省略
    ......
```
以上就是应用本工具的全部内容。
