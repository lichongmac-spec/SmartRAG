from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
    buffer_size=1,
    sentence_split_regex=r"(?<=[。！？])",  # 去掉 \n
    add_start_index=True
)

long_text = """人工智能是计算机科学的一个重要分支，旨在创造能够执行通常需要人类智能的任务的系统。
机器学习是实现人工智能的一种核心方法，它使计算机能够从数据中学习规律。
深度学习则是机器学习的一个子集，它使用多层神经网络来处理复杂的模式识别任务。
与此同时，篮球是一项广受欢迎的运动，NBA联赛汇集了全世界最顶尖的篮球运动员。
在篮球比赛中，球员们通过运球、传球和投篮来争夺分数，团队合作至关重要。"""

# 预处理：去掉换行
long_text = long_text.replace("\n", " ").strip()

chunks = text_splitter.split_text(long_text)

print(f"✅ 原始文本长度: {len(long_text)} 字符")
print(f"✅ 生成块数: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    print(f"--- 块 {i} ---")
    print(chunk)
    print()