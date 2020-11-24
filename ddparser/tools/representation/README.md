## 基于GAT网络应用依存句法分析特征
最近几年图神经网络(GNN)越来越多被应用到NLP任务，而DDParser可以输出句子的依存句法分析树，由于树是图的一种特例，那么很自然的可以将GNN应用在依存句法分析的表示上。<br>
### 实现原理
本工具提出一种方法可以快速将依存句法分析特征应用到下游网络。
* 首先，利用DDParser对用户原始数据处理，得到依存分析结果。
* 其次，修改原始模型的Dataloader，利用本工具提供的接口得到文本的邻接矩阵$\mathrm{A}$。
* 然后，在模型得到句子的表示$\mathrm{X}$后，修改模型结构引入GAT网络，将$\mathrm{A}$和$\mathrm{X}$作为GAT网络输入得到包含句子结构信息的句子表示$\bar{X}$。 
* 最后，对$\bar{X}$应用可分为两种方式。
  - 1.将$\bar{X}$和$\mathrm{X}$拼接后应用到下游网络。
  - 2.通过本工具提供的接口获取句子核心词表示，将核心词的表示应用到下游网络。

经过实验，发现本工具可以有效提升相似度、事件抽取等任务效果。

### 示例
为了方便用户快速使用本工具，我们基于ERNIE1.0模型和LCQMC数据集提供了一个示例。
用户需要自行下载[ERNIE1.0](https://github.com/PaddlePaddle/ERNIE)模型和[LCQMC](http://icrc.hitsz.edu.cn/info/1037/1146.htm)数据集，并通过`preprocess_data.py`对数据预处理。<br>
```python
cd demo/LCQMC
python preprocess_data.py
```
在准备好模型和数据后，用户需要修改demo/run_demo.sh中MODEL_PATH和TASK_DATA_PATH字段指明模型和数据的路径，再通过下列命令即可运行示例。
```python
sh run_demo.sh
```
我们与基线模型做了效果对比，结果如下：

| 模型                    | dev acc | test acc |
| :---------------------- | :------ | :------- |
| ERNIE 1.0 Base          | 89.7    | 87.4     |
| ERNIE 1.0 Base + 本工具 | --      | --       |