from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """检索增强生成技术结合了信息检索与大语言模型。
它能够有效减少模型产生幻觉的问题。
RAG系统的核心步骤包括文档加载、文本切分、向量化和检索生成。
滑动窗口策略可以有效保留上下文信息。"""

# 使用较小的块大小和较大的重叠，模拟滑动窗口效果
splitter = RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=20  # 高重叠率
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks, 1):
    print(f"块 {i}: {chunk}")
