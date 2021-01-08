## 基于句法分析的显式结构表示工具
为了方便用户使用依存句法分析，我们提供了基于句法分析的显示结构表示工具。<br>

### 示例
本工具提供两种粒度的结构化信息抽取接口，如下所示。

```python
>>> from ddparser import DDParser
>>> from extract import FineGrainedInfo
>>> from extract import CoarseGrainedInfo
>>> ddp = DDParser(encoding_model='transformer')
>>> text = ["百度是一家高科技公司"]
>>> ddp_res = ddp.parse(text)
>>> print("依存分析结果:", ddp_res)
依存分析结果: [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
>>> # 细粒度
>>> fine_info = FineGrainedInfo(ddp_res[0])
>>> print("细粒度：", fine_info.parse())
细粒度： [(('百度', '是', '公司'), 'SVO'), (('一家', '公司'), 'ATT_N'), (('高科技', '公司'), 'ATT_N')]
>>> # 粗粒度
>>> coarse_info = CoarseGrainedInfo(ddp_res[0])
>>> print("粗粒度：", coarse_info.parse())
粗粒度： [(('百度', '是', '一家高科技公司'), 'SVO'), (('一家', '公司'), 'ATT_N'), (('高科技', '公司'), 'ATT_N')]
```
### 标签含义
|  标签  |              数据格式              | 实例                  | 说明                                     |
| :----: | :--------------------------------: | :-------------------- | :--------------------------------------- |
| S_V_O  |         (主语, 谓语, 宾语)         | (宝宝, 吃, 奶粉)      | 主谓宾结构                               |
| V_CMP  |           (核心词, 补语)           | (吃, 多久)            | 动补结构，核心词多是动词                 |
| ADV_V  |          (修饰词, 核心词)          | (不能, 吃) (可以, 吃) | 动词及其修饰词，核心词多是动词           |
| ATT_N  |          (修饰词, 核心词)          | (中国, 首都)          | 名词及其修饰词，核心词多是名词           |
|   F    |          (核心词, 方位词)          | (5点, 前)             | 方位词限定核心词，表时间前后、地理位置等 |
|  DOB   | (主谓语, 谓语, 间接宾语, 直接宾语) | (我, 送, 她, 书)      | 双宾语结构                               |
| Phrase |              (核心词)              |                       | 一个词作为整体                           |
