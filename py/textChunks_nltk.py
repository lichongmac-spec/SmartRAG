# ==================== 依赖安装 ====================
# uv pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn

import re
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# ==================== 1. 中文正则分句 ====================
def split_sentences(text):
    """按中文标点和换行切分句子"""
    sentences = re.split(r'(?<=[。！？\n])', text)
    return [s.strip() for s in sentences if s.strip()]

# 准备文档：每个话题扩写为 10 句，共 20 句，保证聚类稳定性
document = """人工智能是计算机科学的重要分支。
机器学习是人工智能的核心方法。
深度学习使用多层神经网络。
自然语言处理让计算机理解人类语言。
计算机视觉使机器能够识别图像。
强化学习通过奖励机制训练智能体。
神经网络是深度学习的基础模型。
监督学习需要标注数据来训练模型。
无监督学习可以从数据中发现隐藏结构。
迁移学习能将知识从一个任务迁移到另一个任务。
篮球是一项广受欢迎的运动。
NBA汇集了全世界顶尖的运动员。
团队合作在篮球比赛中至关重要。
三分球是现代篮球的重要得分手段。
防守策略直接影响比赛胜负。
篮球运动能够锻炼身体协调性。
控球后卫负责组织球队进攻。
中锋通常在篮下得分和抢篮板。
快攻是篮球比赛中常见的得分方式。
篮球比赛分为四节，每节十二分钟。"""

sentences = split_sentences(document)
print(f"✅ 句子数量: {len(sentences)}")

# ==================== 2. 自定义 UMAP（适配中等样本） ====================
umap_model = UMAP(
    n_neighbors=5,        # 样本数 20，n_neighbors 取 5 较安全
    n_components=5,       # 保留更多维度，避免信息过度损失
    metric='cosine',
    random_state=42
)

# ==================== 3. 自定义 HDBSCAN（自动发现主题） ====================
hdbscan_model = HDBSCAN(
    min_cluster_size=3,   # 最小簇大小，至少 3 个句子成簇
    min_samples=2,        # 核心距离的邻居数，小于样本数
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# ==================== 4. 初始化 BERTopic（中文嵌入模型） ====================
topic_model = BERTopic(
    embedding_model="BAAI/bge-small-zh-v1.5",  # 中文优化嵌入模型
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
)

# ==================== 5. 拟合模型 ====================
topics, probs = topic_model.fit_transform(sentences)

# ==================== 6. 输出结果 ====================
print("\n📊 主题分配结果：")
for i, (sentence, topic) in enumerate(zip(sentences, topics), 1):
    print(f"  句子 {i:2d} [主题 {topic}]: {sentence}")

print("\n📌 主题关键词：")
topic_info = topic_model.get_topic_info()
print(topic_info[['Topic', 'Count', 'Name']])

# ==================== 7. 按主题合并为块 ====================
print("\n📦 按主题合并的块：")
for topic_id in sorted(set(topics)):
    if topic_id == -1:
        continue  # 跳过噪声
    topic_sentences = [s for s, t in zip(sentences, topics) if t == topic_id]
    block = " ".join(topic_sentences)
    print(f"\n--- 主题 {topic_id} 块 ---")
    print(block[:120] + "...")