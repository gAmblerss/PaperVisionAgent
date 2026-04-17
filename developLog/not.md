#  开发日志
##  1.RAG返回不对
###  1）chunk切分不好：
减小chunk_size,增大overlap，按结构切分
###  2）检查embedding模型
目前使用的"BAA/bge-small-zh-v1.5" 还行，可以进行query改写
###  3）调检索参数
k值太大太小，太大会导致召回很多相似但不精确的段落，太小就没召回
### 4）尝试MMR
```
retriever = vectorstore.as_triever(
    search_type = "mmr",
    search_kwags = {"k" = 5,"fetch_k":12}
)
```
作用：
*     先多取一些候选
*     在相关性和多样性之间做平衡
### 5）加强metadata
先做metada过滤，再做向量检索
### 6）做混合检索而不是只靠向量
先做关键词过滤，再做向量检索