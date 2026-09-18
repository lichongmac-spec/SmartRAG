# uv pip install semantic-text-splitter tokenizers

from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

# 1. 加载分词器
tokenizer = Tokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
tokenizer.no_truncation()

# 2. 创建分块器
splitter = TextSplitter.from_huggingface_tokenizer(
    tokenizer,
    capacity=150,
    overlap=20
)

# 3. 待分块文本
long_text = """人工智能是计算机科学的一个重要分支，旨在创造能够执行通常需要人类智能的任务的系统。
机器学习是实现人工智能的一种核心方法，它使计算机能够从数据中学习规律。
深度学习则是机器学习的一个子集，它使用多层神经网络来处理复杂的模式识别任务。
与此同时，篮球是一项广受欢迎的运动，NBA联赛汇集了全世界最顶尖的篮球运动员。
在篮球比赛中，球员们通过运球、传球和投篮来争夺分数，团队合作至关重要。"""

# 4. 执行分块
chunks = splitter.chunks(long_text)

# 5. 输出结果
print(f"✅ 原始文本长度: {len(long_text)} 字符")
print(f"✅ 生成块数: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    print(f"--- 块 {i} (长度: {len(chunk)} 字符) ---")
    print(chunk)
    print()