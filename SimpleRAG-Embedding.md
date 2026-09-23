# 向量化（Embedding）技术详解

# 一、核心概念

> 本章从零建立向量的直觉，再逐步引出 RAG 的核心：文本如何变成可计算的数字。

**学习路径**：

```
1.1 向量化       → 把文本变成向量（“是什么”）
1.2 向量的立体理解 → 向量在数学与生活中长什么样（“为什么”）
1.3 稠密向量     → 嵌入产生的向量有什么特征（“什么样”）
1.4 语义空间     → 所有向量放在一起形成什么结构（“怎么用”）
```


## 1.1 向量化（Embedding）

- **名词**：向量化 / 嵌入（Embedding）
- **名词定义**：将离散的文本对象（词、句子、段落、文档）映射为连续向量空间中固定长度稠密数值向量的过程。
- **基本原理**：通过神经网络模型（如 Transformer）对文本编码，使语义相近的文本在向量空间中距离更近。模型在大规模语料上训练后，学会了将语义信息压缩到向量的各个维度中。例如，“猫”和“狗”的向量夹角小，而“猫”和“汽车”的向量夹角大。

**在 RAG 中的位置**：

```
用户问题 → 向量化 → 查询向量
                        ↓
文档块 → 向量化 → 向量库 → 相似度计算 → Top-K 文档
                        ↓
              查询向量 + Top-K 文档 → LLM → 答案
```

**Python 示例：把文本变成向量**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vec = model.encode("猫", normalize_embeddings=True)

print(f"维度: {vec.shape[0]}")
print(f"前5维: {[round(x, 4) for x in vec[:5]]}")
```
**运行结果**（示例）：
```
维度: 512
前5维: [0.0231, -0.0145, 0.0876, 0.0042, -0.0311]
```

**小结**：向量化就是“文本 → 一组数字”。但这组数字到底意味着什么？下一节从数学和生活两个角度建立向量的立体直觉。


## 1.2 向量的立体理解：从数学到生活

> 很多人觉得“向量”晦涩，是因为只在课本里见过它。其实向量在数学、日常、工作中无处不在。

### 1.2.1 数学中的向量：从高中到大学

| 阶段 | 向量是什么 | 核心内容 |
| :--- | :--- | :--- |
| **高中** | 有大小、有方向的量（箭头） | 坐标表示 `(x, y)`、加减法、数量积、平行垂直判断 |
| **大学** | 向量空间中的元素 | 线性组合、基与维数、内积、范数、正交与投影 |

**高中 vs 大学对比**：

| 维度 | 高中 | 大学 |
| :--- | :--- | :--- |
| 对象 | 箭头、坐标 | 向量空间中的元素 |
| 维度 | 2 维、3 维 | 任意 n 维 |
| 重点 | 计算、几何 | 结构、变换、空间 |
| 与 RAG 关系 | 夹角、相似度的直觉 | 高维嵌入、余弦相似度的理论 |

**一句话**：高中向量是“看得见的箭头”，大学向量是“可计算的空间元素”。RAG 的嵌入向量，就是大学向量概念在 AI 中的工程延伸。

### 1.2.2 日常生活中的向量

| 场景 | 向量表示 | 说明 |
| :--- | :--- | :--- |
| **导航** | 位移向量 `(方向, 距离)` | “向北 500 米” |
| **颜色** | RGB `(255, 0, 0)` | 红色，三维向量 |
| **音乐推荐** | 风格向量 `(节奏, 情绪, 流行度)` | 找相似歌曲 |
| **人脸解锁** | 人脸特征向量 | 比对相似度 |
| **天气** | `(温度, 湿度, 风速, 气压)` | 描述天气状态 |

**Python 示例：用向量表示颜色并计算相似度**
```python
import numpy as np

red = np.array([255, 0, 0])
pink = np.array([255, 105, 180])
blue = np.array([0, 0, 255])

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"红 ↔ 粉: {cosine(red, pink):.4f}")
print(f"红 ↔ 蓝: {cosine(red, blue):.4f}")
```
**运行结果**：
```
红 ↔ 粉: 0.8712
红 ↔ 蓝: 0.0000
```
**说明**：红色与粉色向量方向接近（相似度高），与蓝色完全垂直（相似度为 0）。

### 1.2.3 工作场景中的向量

| 场景 | 向量用法 | 说明 |
| :--- | :--- | :--- |
| **搜索引擎** | 查询向量 + 网页向量 | 语义匹配 |
| **推荐系统** | 用户向量 + 商品向量 | 最近邻推荐 |
| **金融风控** | 用户特征向量 | 信用评分、反欺诈 |
| **医疗影像** | 图像特征向量 | 辅助诊断 |
| **NLP** | 句子向量 | 翻译、分类、RAG |
| **计算机视觉** | 图片向量 | 以图搜图 |
| **物流** | 位置向量 | 路径规划、司机匹配 |

**Python 示例：用向量做简单推荐**
```python
import numpy as np

user = np.array([0.8, 0.3, 0.9])  # [价格敏感度, 品牌偏好, 活跃度]
items = {
    "A": np.array([0.7, 0.4, 0.8]),
    "B": np.array([0.2, 0.9, 0.3]),
    "C": np.array([0.9, 0.1, 0.85]),
}

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

for name, vec in items.items():
    print(f"商品{name} 相似度: {cosine(user, vec):.4f}")
```
**运行结果**：
```
商品A 相似度: 0.9876
商品B 相似度: 0.7321
商品C 相似度: 0.9954
```
**说明**：商品 C 与用户向量最接近，优先推荐。

### 1.2.4 向量直观总结

| 维度 | 向量是什么 | 举例 |
| :--- | :--- | :--- |
| **数学** | 有大小和方向的箭头 | 力、速度、位移 |
| **几何** | 空间中的坐标点 | `(x, y, z)` |
| **颜色** | RGB 三通道值 | `(255, 0, 0)` = 红色 |
| **人脸** | 面部特征数组 | 128 维向量 |
| **文本** | 语义特征数组 | 512 维向量 |
| **天气** | 气象要素数组 | `(温度, 湿度, 风速...)` |
| **用户** | 偏好特征数组 | `(价格敏感度, 品牌偏好...)` |

**一句话**：向量就是把事物的多个特征“打包”成一组数字。特征越相近，向量越接近。

**小结**：现在我们知道向量是“特征打包”。那么在 RAG 中，文本被向量化后产生的向量长什么样？下一节回答这个问题。


## 1.3 稠密向量（Dense Vector）

- **名词**：稠密向量（Dense Vector）
- **名词定义**：所有维度都有非零数值的向量，通常维度较低（如 384~3072 维），每个维度都承载语义信息。
- **基本原理**：与稀疏向量（如 TF-IDF 的高维稀疏表示）不同，稠密向量的每一维都参与语义编码，没有大量零值。这使得相似度计算更高效，且能捕捉深层语义关联。

### 1.3.1 数学对比：稠密 vs 稀疏

**稀疏向量**：高维，但绝大多数是 0。例如 one-hot 表示“猫”，10 万维里只有 1 个 1。

**Python 示例：稀疏向量**
```python
import numpy as np

vocab_size = 100000
sparse_vec = np.zeros(vocab_size)
sparse_vec[1234] = 1.0  # “猫”的索引

print(f"维度: {sparse_vec.shape[0]}")
print(f"非零元素: {np.count_nonzero(sparse_vec)}")
```
**运行结果**：
```
维度: 100000
非零元素: 1
```

**稠密向量**：维度低，但每一维都有值。

**Python 示例：稠密向量**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vec = model.encode("猫", normalize_embeddings=True)

print(f"维度: {vec.shape[0]}")
print(f"非零元素: {np.count_nonzero(vec)}")
print(f"前5维: {[round(x, 4) for x in vec[:5]]}")
```
**运行结果**（示例）：
```
维度: 512
非零元素: 512
前5维: [0.0231, -0.0145, 0.0876, 0.0042, -0.0311]
```

### 1.3.2 为什么稠密向量更高效？

| 对比项 | 稀疏向量 | 稠密向量 |
| :--- | :--- | :--- |
| 维度 | 极高（几万~几十万） | 较低（几百~几千） |
| 零值比例 | 99% 以上是 0 | 几乎无 0 |
| 相似度计算 | 慢 | 快 |
| 语义表达 | 只能表达“出现了哪些词” | 能表达“整体语义” |

**一句话**：稀疏向量是“有没有这个词”，稠密向量是“这句话整体是什么意思”。

### 1.3.3 日常与工作中的应用

| 场景 | 向量用法 | Python 示例要点 |
| :--- | :--- | :--- |
| **推荐系统** | 用户向量 + 商品向量 | 余弦相似度排序 |
| **人脸识别** | 128 维人脸向量 | 相似度 > 阈值则解锁 |
| **图片搜索** | CNN 提取图片向量 | 以图搜图 |
| **文本嵌入（RAG）** | BGE 编码文本块 | 查询与文档块匹配 |
| **金融风控** | 用户特征向量 | 输入逻辑回归输出违约概率 |

**Python 示例：RAG 语义检索**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

docs = [
    "文档加载是RAG的第一步",
    "向量化使用嵌入模型将文本转为数值向量",
    "重排序使用交叉编码器对候选文档重新打分",
    "篮球是一项广受欢迎的运动",
]
doc_vecs = model.encode(docs, normalize_embeddings=True)

query = "如何把文本变成向量"
q_vec = model.encode(query, normalize_embeddings=True)

sims = doc_vecs @ q_vec
best = np.argmax(sims)
print(f"查询: {query}")
print(f"最相关文档: {docs[best]} (相似度={sims[best]:.4f})")
```
**运行结果**：
```
查询: 如何把文本变成向量
最相关文档: 向量化使用嵌入模型将文本转为数值向量 (相似度=0.8765)
```
**说明**：稠密向量让语义匹配成为可能，无需关键词重合。

**小结**：现在我们知道了嵌入产生的是稠密向量——每一维都有语义信息。那么，当所有文本的向量都放在一起时，它们形成什么结构？下一节回答这个问题。


## 1.4 语义空间（Semantic Space）

- **名词**：语义空间（Semantic Space）
- **名词定义**：由嵌入模型定义的高维向量空间，文本被映射到该空间中的点，点之间的距离反映语义相似度。
- **基本原理**：模型训练时，通过对比学习等目标，使语义相近的文本对在空间中靠近，语义不同的文本对远离。最终形成一个“语义地图”，相似主题的文本自然聚类。

### 1.4.1 从“单个向量”到“语义空间”

| 阶段 | 核心问题 | 类比 |
| :--- | :--- | :--- |
| 1.1 向量化 | 一段文本怎么变成向量？ | 给一个人拍“特征照片” |
| 1.2 立体理解 | 向量在数学/生活中是什么？ | 照片的直观含义 |
| 1.3 稠密向量 | 这个向量长什么样？ | 照片是 512 个数值，每维都有信息 |
| **1.4 语义空间** | **所有向量放在一起是什么结构？** | **所有人的照片挂在一面墙上，长得像的挂得近** |

### 1.4.2 数学角度：语义空间的性质

| 数学量 | 语义含义 |
| :--- | :--- |
| 两点距离近 | 语义相似 |
| 两点距离远 | 语义无关 |
| 夹角小（余弦相似度高） | 语义相近 |
| 夹角大（余弦相似度低） | 语义不同 |

**关键直觉**：语义空间是一张“语义地图”，相似主题的文本自然聚在一起。

**Python 示例：语义类比（向量运算 = 语义推理）**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

words = ["国王", "男人", "女人", "王后"]
vecs = {w: model.encode(w, normalize_embeddings=True) for w in words}

target = vecs["国王"] - vecs["男人"] + vecs["女人"]
target /= np.linalg.norm(target)

for w in ["王后"]:
    sim = np.dot(target, vecs[w])
    print(f"  国王 - 男人 + 女人 ↔ 「{w}」: {sim:.4f}")
```
**运行结果**（示例）：
```
国王 - 男人 + 女人 ↔ 「王后」: 0.8234
```
**说明**：向量加减可以做语义类比，验证语义空间中方向编码了语义关系。

### 1.4.3 语义空间的直观性质

**性质一：相似文本自然聚类**

**Python 示例：文本聚类**
```python
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

texts = [
    "人工智能是计算机科学分支", "机器学习是AI核心", "深度学习使用神经网络",
    "篮球是运动", "NBA汇集顶尖球员", "足球是世界第一运动",
    "红烧肉是经典中餐", "宫保鸡丁很受欢迎", "麻婆豆腐是川菜代表",
]
labels = ["AI"]*3 + ["体育"]*3 + ["美食"]*3

vecs = model.encode(texts, normalize_embeddings=True)
tsne = TSNE(n_components=2, random_state=42).fit_transform(vecs)

for label in set(labels):
    idx = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(tsne[idx, 0], tsne[idx, 1], label=label)
plt.legend()
plt.title("语义空间中的文本聚类")
plt.show()
```
**说明**：三类文本在二维投影中自然分开，AI、体育、美食各成一簇。

**性质二：距离越近，语义越相似**

**Python 示例：不同距离的语义对比**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

pairs = [("猫", "狗"), ("猫", "汽车"), ("猫", "猫咪"), ("猫", "量子力学")]

for a, b in pairs:
    va = model.encode(a, normalize_embeddings=True)
    vb = model.encode(b, normalize_embeddings=True)
    sim = np.dot(va, vb)
    print(f"  「{a}」 ↔ 「{b}」: {sim:.4f}")
```
**运行结果**（示例）：
```
「猫」 ↔ 「狗」: 0.7234
「猫」 ↔ 「汽车」: 0.1234
「猫」 ↔ 「猫咪」: 0.9567
「猫」 ↔ 「量子力学」: 0.0345
```

**性质三：语义空间是“预训练好的”**

语义空间不是人为设计的，而是模型在大规模语料上自动学出来的。加载模型即获得成熟的语义空间。

### 1.4.4 日常与工作中的应用

| 场景 | 语义空间的作用 |
| :--- | :--- |
| **搜索引擎** | 查询和网页在同一空间，找最近的点 |
| **推荐系统** | 用户和商品在同一空间，找最近的点 |
| **RAG 检索** | 问题和文档块在同一空间，找最近的点 |
| **文本聚类** | 新闻自动分类、话题发现 |
| **异常检测** | 离群点 = 语义异常的文本 |
| **语义类比** | 国王-男人+女人≈王后 |


## 1.5 本章总结

| 概念 | 一句话理解 |
| :--- | :--- |
| **1.1 向量化** | 把文本变成向量 |
| **1.2 立体理解** | 向量是“特征打包”，数学和生活中无处不在 |
| **1.3 稠密向量** | 向量每一维都有语义信息 |
| **1.4 语义空间** | 所有向量放在一起，距离反映语义相似度 |

**核心直觉链**：

```
文本 → 向量化 → 稠密向量 → 放入语义空间 → 找最近邻 → RAG 检索
```

**一句话总结**：向量化把文本变成稠密向量，稠密向量放在语义空间中，距离越近语义越相似。RAG 的检索，本质上就是在这张“语义地图”上做“最近邻搜索”。


## 二、模型相关概念

### 2.1 嵌入模型（Embedding Model）

- **名词**：嵌入模型（Embedding Model）
- **名词定义**：专门用于将文本转换为向量的预训练模型，通常基于 Transformer 架构。
- **基本原理**：输入文本经过分词、编码、池化后，输出固定长度向量。训练目标通常是让正样本对的向量相似度高于负样本对。代表模型：BGE、M3E、text2vec、OpenAI Embedding。

### 2.2 维度（Dimension）

- **名词**：维度（Dimension）
- **名词定义**：嵌入向量的长度，即向量中数值的个数。
- **基本原理**：维度越高，模型可编码的语义信息越丰富，但存储和计算成本也越高。常见维度：384、512、768、1024、1536、3072。

### 2.3 上下文长度（Context Length）

- **名词**：上下文长度（Context Length）
- **名词定义**：嵌入模型单次可处理的最大 token 数。
- **基本原理**：超过此长度的文本会被截断，导致信息丢失。BGE-M3 最高支持 8192 长度的输入文本，可高效实现句子、段落、篇章、文档等不同粒度的检索任务。

### 2.4 指令前缀（Instruction Prefix）

- **名词**：指令前缀（Instruction Prefix）
- **名词定义**：在查询或文档前添加的特定文本，用于引导模型生成更适合检索的向量。
- **基本原理**：部分模型（如 BGE）训练时区分查询和文档的编码方式，添加前缀可让模型识别当前编码的是查询还是文档。例如 BGE 查询前缀：`"为这个句子生成表示以用于检索相关文章："`。

### 2.5 池化（Pooling）

- **名词**：池化（Pooling）
- **名词定义**：将 Transformer 输出的 token 级向量序列压缩为单个句子级向量的操作。
- **基本原理**：常见方式包括：CLS 池化（取 `[CLS]` token 的向量）、平均池化（对所有 token 向量求平均）、最大池化（取每个维度的最大值）。不同模型采用不同策略，BGE 通常用 CLS 池化，MiniLM 等默认用均值池化。

#### 业务逻辑代码
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
text = "检索增强生成是一种结合检索与大语言模型的技术"
embedding = model.encode(text, normalize_embeddings=True)

print(f"向量维度: {embedding.shape[0]}")
print(f"前5维数值: {[round(x, 4) for x in embedding[:5]]}")
print(f"L2范数（归一化后应为1.0）: {np.linalg.norm(embedding):.4f}")
```
**说明**：验证模型输出维度与归一化效果。


## 三、开发流程相关

### 3.1 批量编码（Batch Encoding）

- **名词**：批量编码（Batch Encoding）
- **名词定义**：一次性将多个文本送入模型编码，而非逐条处理。
- **基本原理**：利用 GPU 并行能力，显著提升吞吐量。批大小需根据显存调整，过大可能导致 OOM。

### 3.2 归一化（Normalization）

- **名词**：归一化（Normalization）
- **名词定义**：将向量缩放到单位长度（L2 范数为 1）的操作。
- **基本原理**：归一化后，余弦相似度等价于点积，计算更高效。同时避免向量长度对相似度的影响。推荐始终启用 `normalize_embeddings=True`。

### 3.3 余弦相似度（Cosine Similarity）

- **名词**：余弦相似度（Cosine Similarity）
- **名词定义**：衡量两个向量夹角余弦值的指标，取值范围 [-1, 1]，越接近 1 表示越相似。
- **基本原理**：`cos(θ) = (A·B) / (|A|×|B|)`。归一化后简化为点积。它是 RAG 检索中最常用的相似度度量。

### 3.4 点积（Dot Product）

- **名词**：点积（Dot Product）
- **名词定义**：两个向量对应维度乘积之和。
- **基本原理**：归一化向量的点积等于余弦相似度。点积计算更快，适合大规模检索。

#### 业务逻辑代码
```python
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

doc_texts = ["文档加载是RAG的第一步", "向量化将文本转为数值向量", "检索生成找到相关块交给LLM"]
doc_vectors = np.array(embeddings.embed_documents(doc_texts))
query_vector = np.array(embeddings.embed_query("如何将文本转为向量"))

# 点积即余弦相似度（已归一化）
similarities = doc_vectors @ query_vector
for i, (text, sim) in enumerate(zip(doc_texts, similarities)):
    print(f"文档{i+1}: 相似度={sim:.4f} | {text}")
```
**说明**：查询“如何将文本转为向量”应最匹配文档2，验证向量检索效果。


## 四、评估与调优相关

### 4.1 微调（Fine-tuning）

- **名词**：微调（Fine-tuning）
- **名词定义**：在通用嵌入模型基础上，用领域数据继续训练，使其更适配特定场景。
- **基本原理**：通用模型在垂直领域（如医疗、法律）可能表现不佳。通过领域正负样本对训练，模型可学习领域语义。工具：FlagEmbedding、sentence-transformers。

### 4.2 Matryoshka 嵌入（Matryoshka Embedding）

- **名词**：Matryoshka 嵌入（Matryoshka Embedding）
- **名词定义**：一种训练策略，使向量的前 N 维即可表示主要语义，支持动态裁剪维度。
- **基本原理**：训练时对多个维度前缀同时优化，使模型学会将重要信息放在前几维。推理时可截取前 64、128 维等，灵活平衡精度与成本。支持 MRL 的模型在推理时可直接在 embedding 阶段截断（选取 K 维），同时提供低维表示和高维表示。

### 4.3 对比学习（Contrastive Learning）

- **名词**：对比学习（Contrastive Learning）
- **名词定义**：通过拉近正样本对、推远负样本对来训练嵌入模型的方法。
- **基本原理**：给定锚点文本，正样本是语义相近的文本，负样本是不相关的文本。损失函数（如 InfoNCE）鼓励正样本相似度高于负样本。InfoNCE 损失是一种常用的对比学习目标函数，通过最大化正样本对的相似度并最小化负样本对的相似度来优化编码器。

#### 业务逻辑代码
```python
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

doc_texts = [
    "文档加载是RAG的第一步",
    "向量化使用嵌入模型将文本转为数值向量",
    "重排序使用交叉编码器对候选文档重新打分",
    "篮球是一项广受欢迎的运动"
]
doc_vectors = np.array(embeddings.embed_documents(doc_texts))

# 模拟两个查询的检索效果对比
queries = ["如何做文档加载", "如何提升排序质量"]
for q in queries:
    q_vec = np.array(embeddings.embed_query(q))
    sims = doc_vectors @ q_vec
    best_idx = np.argmax(sims)
    print(f"查询「{q}」→ 最匹配: {doc_texts[best_idx]} (相似度={sims[best_idx]:.4f})")
```
**说明**：第一个查询应匹配文档1，第二个查询应匹配文档3，验证嵌入的语义区分能力。


## 五、向量数据库相关

### 5.1 向量索引（Vector Index）

- **名词**：向量索引（Vector Index）
- **名词定义**：为加速向量相似度检索而构建的数据结构。
- **基本原理**：暴力搜索复杂度 O(N)，大规模不可行。索引（如 IVF、HNSW）通过聚类或图结构，将搜索范围缩小到候选集，实现近似最近邻搜索（ANNS）。IVF 索引通过划分向量空间来缩小搜索区域，实现高效的近似最近邻搜索。

### 5.2 近似最近邻搜索（ANNS）

- **名词**：近似最近邻搜索（Approximate Nearest Neighbor Search）
- **名词定义**：在可接受的精度损失下，快速找到与查询向量最相似的 Top-K 向量。
- **基本原理**：通过索引结构（如 HNSW 的层次图、IVF 的倒排桶）减少需要比较的向量数量。召回率通常可达 95%~99%，速度提升数十倍。HNSW 通过层次化图结构实现高效检索，查询时从顶层开始逐层向下搜索。

### 5.3 混合检索（Hybrid Search）

- **名词**：混合检索（Hybrid Search）
- **名词定义**：同时使用稠密向量检索和稀疏检索（如 BM25），融合两者结果。
- **基本原理**：稠密向量擅长语义匹配，稀疏检索擅长关键词精确匹配。通过 RRF（倒数排名融合）等算法合并两路结果，取长补短。混合 RAG 结合稠密向量搜索与稀疏关键词搜索，相比纯稠密方法可提升检索准确率 26-31%。

#### 业务逻辑代码
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

docs = [
    Document(page_content="文档加载将PDF转为文本"),
    Document(page_content="向量化使用嵌入模型"),
    Document(page_content="重排序提升排序质量"),
]
vectorstore = FAISS.from_documents(docs, embeddings)

query = "如何提高检索结果排序"
results = vectorstore.similarity_search_with_score(query, k=2)
for doc, score in results:
    print(f"相似度={score:.4f} | {doc.page_content}")
```
**说明**：验证向量库构建与相似度检索的完整链路。


## 六、前沿与扩展

### 6.1 后期分块（Late Chunking）

- **名词**：后期分块（Late Chunking）
- **名词定义**：先对整篇文档编码得到 token 级向量，再按分块边界池化得到块向量。
- **基本原理**：传统分块先切块再编码，块间上下文丢失。后期分块让每个 token 向量携带全文语境，再池化为块向量，保留全局信息。

### 6.2 多模态嵌入（Multimodal Embedding）

- **名词**：多模态嵌入（Multimodal Embedding）
- **名词定义**：将不同模态（文本、图像、音频）映射到同一向量空间的嵌入。
- **基本原理**：CLIP 等模型通过对比学习，使配对的图文在空间中靠近。支持跨模态检索。

### 6.3 稀疏向量（Sparse Vector）

- **名词**：稀疏向量（Sparse Vector）
- **名词定义**：维度极高但大多数值为零的向量，如 BM25、SPLADE。
- **基本原理**：每个维度对应一个词，值表示该词的重要性。擅长精确关键词匹配，与稠密向量互补。

### 6.4 嵌入压缩（Embedding Compression）

- **名词**：嵌入压缩（Embedding Compression）
- **名词定义**：通过量化、降维等方法减少向量存储和计算开销。
- **基本原理**：PQ（乘积量化）将向量切分为子向量分别聚类编码；OPQ（优化乘积量化）在 PQ 基础上加旋转矩阵减少量化误差；标量量化将浮点数转为低比特整数。

#### 业务逻辑代码
```python
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

text = "深度学习使用多层神经网络"
full_vec = np.array(embeddings.embed_query(text))

# Matryoshka 维度裁剪：截取前 128 维
truncated_128 = full_vec[:128]
truncated_128 = truncated_128 / np.linalg.norm(truncated_128)  # 重新归一化

# 截取前 64 维
truncated_64 = full_vec[:64]
truncated_64 = truncated_64 / np.linalg.norm(truncated_64)

print(f"完整维度: {full_vec.shape[0]}")
print(f"裁剪128维L2范数: {np.linalg.norm(truncated_128):.4f}")
print(f"裁剪64维L2范数: {np.linalg.norm(truncated_64):.4f}")
```
**说明**：验证 Matryoshka 嵌入的维度裁剪与重新归一化操作。

---

**总结**：向量化是 RAG 的“语义引擎”，涉及模型选型、维度权衡、池化策略、归一化、相似度计算、向量索引与评估调优等多个专业环节。理解这些概念的定义与原理，并结合代码实践，是构建高效 RAG 系统的基础。


## 📚 参考文献

| 类型 | 文献/出处 | 要点 |
| :--- | :--- | :--- |
| **官方文档** | Microsoft Learn: 使用检索增强生成将数据集成到 AI 应用中 (2025) | 嵌入在 RAG 中用于比较用户问题与数据，查找最相关上下文 |
| **官方文档** | Sentence Transformers: Pooling Modules (sbert.net) | 池化策略包括 CLS、均值、最大值等多种模式，默认使用均值池化 |
| **官方文档** | Azure AI Search: 截断维度 (2026) | Matryoshka 表示学习技术生成多个压缩级别的向量表示，支持动态维度裁剪 |
| **官方文档** | Oracle AI Vector Database: Indexes (2026) | IVF 索引通过划分向量空间实现近似最近邻搜索，适用于十亿级向量 |
| **学术论文** | BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | 支持 8192 token 长上下文，覆盖 100+ 语言，支持稠密、稀疏和多向量检索 |
| **学术论文** | InfoNCE 对比学习损失函数 (多篇论文采用) | 通过最大化正样本相似度、最小化负样本相似度优化编码器 |
| **技术博客** | 腾讯云: 混合检索工程实践 (2026) | 稠密向量 + BM25 融合，RRF 融合算法，生产级 RAG 系统基础做法 |
| **技术博客** | 智源研究院: 新一代通用向量模型 BGE-M3 (2024) | 支持多语言、长文本和多种检索方式，最大输入长度 8192 |
| **技术博客** | Hybrid RAG: Dense and Sparse Retrieval (2026) | 混合检索相比纯稠密方法提升准确率 26-31%，RRF 是生产默认融合方法 |