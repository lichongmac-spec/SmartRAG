"""
分块策略选择与评估：对比多种分块策略的检索效果
评估指标：Hit Rate@K、MRR、Recall@K
改进点：增加干扰文档 + 嵌入模型检索 + 语义查询
"""
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# ==================== 1. 文档：核心内容 + 干扰内容 ====================
document = """检索增强生成（RAG）是一种结合信息检索与大语言模型的技术。
文档加载是RAG的第一步，将PDF、Word等格式转为文本。
文本切分将长文档拆分为语义完整的块，是检索精度的关键环节。
向量化使用嵌入模型将文本转为数值向量。
检索生成根据用户问题找到相关块并交给LLM生成答案。
混合检索结合向量检索和关键词检索的优势，提升召回率。
重排序使用交叉编码器对候选文档重新打分，提升排序质量。
多粒度索引同时建立句子、段落、章节三级索引。
后期分块先整篇编码再切块，保留全局语境信息。

机器学习是人工智能的核心方法，通过数据训练模型。
深度学习使用多层神经网络处理复杂模式。
自然语言处理让计算机理解人类语言。
计算机视觉使机器能够识别图像。
强化学习通过奖励机制训练智能体。
数据库索引用于加速数据查询。
缓存策略可以提升系统响应速度。
分布式系统通过多节点协同处理任务。
消息队列用于异步通信和解耦。
负载均衡将请求分发到多个服务器。
微服务架构将应用拆分为独立服务。
容器化技术简化了应用部署流程。"""

# ==================== 2. 测试集：语义查询（非字面匹配） ====================
test_set = [
    {"query": "如何让检索结果更准确", "expected_keyword": "重排序"},
    {"query": "怎样保留文档的全局语境", "expected_keyword": "后期分块"},
    {"query": "如何提升召回率", "expected_keyword": "混合检索"},
    {"query": "如何建立多层次的索引", "expected_keyword": "多粒度"},
    {"query": "文档处理的第一步是什么", "expected_keyword": "文档加载"},
]

# ==================== 3. 加载嵌入模型 ====================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# ==================== 4. 定义多种分块策略 ====================
strategies = {
    "字符分块(40)": CharacterTextSplitter(chunk_size=40, chunk_overlap=5, separator="\n"),
    "递归(25)": RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=5),
    "递归(50)": RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10),
    "递归(100)": RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20),
}

# ==================== 5. 评估函数 ====================
def evaluate(chunks, test_set, k=3):
    """计算 Hit Rate@K、MRR、Recall@K"""
    # 嵌入所有块
    chunk_vecs = np.array(embeddings.embed_documents(chunks))
    
    hit_count, rr_sum, recall_sum = 0, 0.0, 0.0
    
    for item in test_set:
        # 查询嵌入
        q_vec = np.array(embeddings.embed_query(item["query"]))
        # 余弦相似度排序
        sims = chunk_vecs @ q_vec
        top_k_idx = np.argsort(sims)[::-1][:k]
        
        # 找到第一个包含期望关键词的块排名
        rank = None
        for i, idx in enumerate(top_k_idx):
            if item["expected_keyword"] in chunks[idx]:
                rank = i + 1
                break
        
        # 统计指标
        if rank is not None:
            hit_count += 1
            rr_sum += 1.0 / rank
        
        # Recall@K：所有相关块中被检索到的比例
        all_relevant = [i for i, c in enumerate(chunks) if item["expected_keyword"] in c]
        retrieved_relevant = [i for i in top_k_idx if item["expected_keyword"] in chunks[i]]
        if all_relevant:
            recall_sum += len(retrieved_relevant) / len(all_relevant)
    
    n = len(test_set)
    return hit_count / n, rr_sum / n, recall_sum / n

# ==================== 6. 对比各策略 ====================
print(f"{'策略':<14} {'块数':<6} {'Hit@3':<8} {'MRR':<8} {'Recall@3':<10}")
print("-" * 50)

results = []
for name, splitter in strategies.items():
    chunks = splitter.split_text(document)
    hit, mrr, recall = evaluate(chunks, test_set, k=3)
    results.append((name, len(chunks), hit, mrr, recall))
    print(f"{name:<14} {len(chunks):<6} {hit:<8.2f} {mrr:<8.4f} {recall:<10.4f}")

# ==================== 7. 最优策略推荐 ====================
best = max(results, key=lambda x: x[3])  # 按 MRR 排序
print(f"\n🏆 MRR 最高策略: {best[0]} (MRR={best[3]:.4f})")