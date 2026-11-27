# workflow-rag-retriever

给专业文档做问答的检索流程。适合那种问题里带着明确编号的场景，比如设备号、项目号、图号。

仓库里的代码是重写的，数据全是合成的（`EQ-1001` 这种编的设备号）。没有真实客户、真实编号、内网地址或者密钥。想看算法怎么实现的，可以直接往下翻。

[English version](README.en.md)

## 为什么要做这个

一开始就是最普通的 RAG：问题转向量，全库搜。通用问答里够用，一到专业文档就不行。

举个具体的例子。用户问"EQ-1001 的维护周期是多少"。全库向量检索会把所有提到 EQ-1001 的文档都捞回来：维护手册、巡检记录、采购单、报废单。语义上它们都相关，但答案只在维护手册里。最后的结果就是，召回看着挺好看，答出来的东西对不上。

后来加了一层元数据。先从问题里把设备编号抽出来，查这个编号对应哪几份文档，检索只在这几份里做。上面那个例子，搜索范围从全库缩到两三份文档。

再往后发现缩小范围还不够。有些答案埋在大文档的某一节里，切成片之后上下文断了，光靠向量相似度找过去找不着。于是又加了 PageIndex：给每份文档建一棵章节树，去掉正文只留标题和摘要，让模型看完树以后挑出该读哪几节，连页码一起带回来。

两步加完，同一批问题，能用的答案从六成到了九成五。

## 效果怎么看

评测集 300 题，两种方案的对比：

| | 老方案（纯向量检索） | 这套流程 |
|---|---|---|
| 答案能用 | 60% | **95%** |
| 文档召回率 | — | 0.90 |
| 至少命中一份目标文档 | — | 97.3% |

"能用"是这么算的：300 题里完全正确 162 题，部分正确 123 题，加起来 285 题，正好 95%。严格只算"每个数字都对"的话是 54%，但业务上答案基本对、少一两个数的，也算能用。

这套流程真正改变的是下限。老方案六成的能用率意味着十道题里有四道得人工兜底，根本没法上线；九成五是"偶尔补一下"的量级。中间差的这三十几个点，不是调参数调出来的，是检索范围从全库缩到相关文档、再补上章节结构这两步带来的。

召回率 0.90 说明绝大多数问题都能找到目标文档，但文档找到了不代表答案对。评测里有 29 题召回是满分、答案却错了或者只答对一半，其中 10 题直接判错。这部分问题出在抽取和生成，不在检索，这套代码管不到。`evaluation.py` 里专门放了 `perfect_recall_correct_rate` 和 `recall_one_but_not_correct` 两个指标，就是为了别让一个好看的召回率把生成环节的问题盖住。

顺带一提，表格类问题是最难啃的。全部 5 道零召回都出在这一类，而且是同一个问法连续挂了 5 次。这种批量失败查起来比零散错误容易，基本能锁定是索引或者规范化的问题。

## 流程长这样

```
                          用户问题
                              |
                   [1] 抽实体
            schema few-shot + 规则/正则兜底
                              |
                   [2] 查元数据
             实体 -> 文档 ID + 章节树路径
                              |
       +----------------------+-------------------+
       |                                          |
  [3] 摘要检索兜底                          （元数据命中）
   元数据什么都没找到时                            |
       +----------------------+-------------------+
                              |
       [4] 两路并行检索
        全文向量   ||   PageIndex 章节树
      （改写后的问题也作为额外的 query）
                              |
             [5] RRF 融合 -> 去重 -> top_k
                              |
                    [6] 组装结果 + 进度事件
```

每一步都是一个协议，定义在 `ragflow/ports.py`。`ragflow/workflow.py` 只认这些协议，不认具体的服务。换成自己的向量库、元数据接口或者模型，改实现就行，编排不用动。

## 文件都是干嘛的

| 文件 | 作用 |
|---|---|
| `ragflow/entity.py` | 从问题里抽实体。schema few-shot，时间规则、正则兜底、别名和分隔符规范化 |
| `ragflow/metadata.py` | 实体到文档的映射，也就是缩小检索范围那一步 |
| `ragflow/pageindex.py` | 章节树检索，模型挑节点，没模型时用字面匹配 |
| `ragflow/rewrite.py` | 多轮改写和同义词改写 |
| `ragflow/fusion.py` | 分数归一化、RRF 融合、去重 |
| `ragflow/workflow.py` | 上面那六步的编排 |
| `ragflow/evaluation.py` | 召回率、命中率、全命中率、召回满分正确率 |
| `ragflow/progress.py` | 进度事件，失败不影响检索主流程 |
| `ragflow/backends/demo.py` | 离线演示用的假数据和假后端 |

## 跑一下

不用装任何东西，标准库就够。

```bash
python examples/demo.py       # 完整跑一遍检索
python examples/evaluate.py   # 跑评测示例
python run_tests.py           # 21 个测试
```

`examples/demo.py` 的输出大概是这样：

```
Q: What is the maintenance interval for EQ-1001?
entities: ['EQ-1001']
documents: ['DOC-A', 'DOC-B']
  [vector   ] 0.049  The maintenance interval for EQ-1001 is 180 days...
  [pageindex] 0.032  Maintenance interval: Scheduled maintenance cycle is 180 days...
```

装了 pytest 的话，`pytest tests` 也能跑，测试就是普通的 pytest 风格函数。

## 接自己的服务

实现 `ragflow/ports.py` 里对应的协议就行：

```python
from ragflow import WorkflowRetriever
from ragflow.entity import SchemaItem

retriever = WorkflowRetriever.from_llm(
    llm=my_llm,                        # 实现 complete(system, user)
    vector_retriever=my_vector_store,  # 实现 search(...)
    metadata_provider=my_metadata,     # 实现 lookup(entities)
    tree_store=my_tree_store,          # 实现 load(toc_path)
    schema=[SchemaItem("asset", "EQ-1001, EQ-2002")],
    aliases={"EQ 1001": "EQ-1001"},    # 口语写法 -> 标准编号
)
result = retriever.retrieve(RetrieveRequest(query="..."))
```

没配模型也能跑，会退到规则抽取。想先验证流程通不通，用这个就够了。

## 几个实现上的取舍

向量分和章节树的分不在一个量纲上，直接加权平均是在赌。用 RRF 按排名融合，省掉了跨通道调分数的麻烦。

元数据没命中的时候不直接返回空，拿摘要再搜一遍把候选文档捞回来。宁可多查几份，也别让一次编号没对上就把整个问题废掉。

查元数据之前先做规范化。`EQ 1001`、`eq-1001`、`EQ—1001` 都得落到同一个标准编号上，否则元数据那边查不到。

章节树里父子节点都命中时，子节点排在前面。父节点通常只是个标题，子节点才有具体参数。

核心逻辑不依赖任何第三方包。基础设施都在协议后面，换掉不影响算法本身。

## 还没解决的问题

抽取和生成那一段不在这套代码里。前面说的 29 题召回满分但答错，就是那一段的事，文档再准也救不回来。

图片类问题目前最弱，文档命中了图片也可能没被返回。这一块得单独做图注索引和图片回传链路。

表格类问题的检索也不稳，5 次零召回全在这。真要上生产，这个得优先查。

## License

MIT，见 [LICENSE](LICENSE)。
