from langchain_text_splitters import SpacyTextSplitter

# spaCy 需要下载语言模型：python -m spacy download en_core_web_sm
text = """检索增强生成（RAG）是一种结合信息检索与大语言模型的技术。
它能够有效减少模型产生幻觉的问题。
RAG系统的核心步骤包括文档加载、文本切分、向量化和检索生成。
其中，文本切分是决定检索精度的关键环节。"""

# 创建 spaCy 分块器：chunk_size=1000 字符
splitter = SpacyTextSplitter(
    chunk_size=1000,       # 每块最大字符数
    chunk_overlap=100,     # 块间重叠字符数
    separator="\n" ,      # 句子之间的分隔
                            # 句子之间的分隔符
    pipeline="zh_core_web_sm"  # 指定中文模型
)

chunks = splitter.split_text(text)

print(f"✅ 原始文本长度: {len(text)} 字符")
print(f"✅ 生成块数: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    print(f"块 {i}: {chunk}")
