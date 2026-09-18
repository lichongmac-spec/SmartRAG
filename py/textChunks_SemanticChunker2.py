# ==================== 安装依赖 ====================
# uv pip install sentence-transformers numpy scikit-learn

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 1. 句子切分 ====================
def split_into_sentences(text):
    """使用正则表达式将文本切分为句子（支持中文标点）"""
    sentences = re.split(r'(?<=[。！？\n])', text)
    return [s.strip() for s in sentences if s.strip()]

# ==================== 2. 向量化 ====================
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

def embed_sentences(sentences):
    """将每个句子转换为嵌入向量"""
    return model.encode(sentences, normalize_embeddings=True)

# ==================== 3. 计算语义距离（余弦相似度） ====================
def compute_similarities(embeddings):
    """计算相邻句子向量之间的余弦相似度"""
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
    return similarities

# ==================== 4. 断点检测 ====================
def find_breakpoints(similarities, method="percentile", amount=70):
    """根据相似度序列检测语义断裂点"""
    sims = np.array(similarities)

    if method == "percentile":
        # 百分位法：相似度低于第 amount 百分位的位置为断点
        threshold = np.percentile(sims, 100 - amount)  # 注意：相似度越低越可能是断点
        breakpoints = [i for i, s in enumerate(sims) if s < threshold]

    elif method == "standard_deviation":
        # 标准差法：低于 均值 - amount * 标准差 的位置为断点
        mean = np.mean(sims)
        std = np.std(sims)
        threshold = mean - amount * std
        breakpoints = [i for i, s in enumerate(sims) if s < threshold]

    elif method == "interquartile":
        # 四分位距法：低于 Q1 - 1.5 * IQR 的位置为断点
        q1, q3 = np.percentile(sims, [25, 75])
        iqr = q3 - q1
        threshold = q1 - 1.5 * iqr
        breakpoints = [i for i, s in enumerate(sims) if s < threshold]

    else:
        raise ValueError(f"未知的断点检测方法: {method}")

    return breakpoints, threshold

# ==================== 5. 合并块 ====================
def merge_into_chunks(sentences, breakpoints):
    """根据断点将句子合并为语义连贯的块"""
    chunks = []
    start = 0
    for bp in breakpoints:
        chunk = "".join(sentences[start:bp + 1])
        if chunk.strip():
            chunks.append(chunk.strip())
        start = bp + 1
    # 处理最后一个块
    if start < len(sentences):
        chunks.append("".join(sentences[start:]).strip())
    return chunks

# ==================== 完整流程 ====================
text = """人工智能是计算机科学的一个重要分支，旨在创造能够执行通常需要人类智能的任务的系统。
机器学习是实现人工智能的一种核心方法，它使计算机能够从数据中学习规律。
深度学习则是机器学习的一个子集，它使用多层神经网络来处理复杂的模式识别任务。
与此同时，篮球是一项广受欢迎的运动，NBA联赛汇集了全世界最顶尖的篮球运动员。
在篮球比赛中，球员们通过运球、传球和投篮来争夺分数，团队合作至关重要。"""

# 步骤 1：句子切分
sentences = split_into_sentences(text)
print("=== 步骤 1：句子切分 ===")
for i, s in enumerate(sentences):
    print(f"  句子 {i}: {s}")
print()

# 步骤 2：向量化
embeddings = embed_sentences(sentences)
print(f"=== 步骤 2：向量化 ===")
print(f"  共生成 {len(embeddings)} 个向量，每个向量维度: {embeddings[0].shape[0]}")
print()

# 步骤 3：计算语义距离
similarities = compute_similarities(embeddings)
print("=== 步骤 3：计算相邻句子余弦相似度 ===")
for i, sim in enumerate(similarities):
    print(f"  句子{i} ↔ 句子{i+1}: 相似度 = {sim:.4f}")
print()

# 步骤 4：断点检测
breakpoints, threshold = find_breakpoints(similarities, method="percentile", amount=70)
print("=== 步骤 4：断点检测（百分位法） ===")
print(f"  断点阈值 = {threshold:.4f}")
print(f"  检测到的断点索引: {breakpoints}")
for bp in breakpoints:
    print(f"    → 在句子{bp}和句子{bp+1}之间切分")
print()

# 步骤 5：合并块
chunks = merge_into_chunks(sentences, breakpoints)
print("=== 步骤 5：合并块 ===")
for i, chunk in enumerate(chunks, 1):
    print(f"  块 {i}: {chunk}")
    print()
