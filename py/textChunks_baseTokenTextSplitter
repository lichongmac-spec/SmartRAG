from langchain_text_splitters import TokenTextSplitter

# 使用 OpenAI 的 cl100k_base 分词器，每块 20 个词元，重叠 5 个词元
splitter = TokenTextSplitter(
    chunk_size=20,
    chunk_overlap=5,
    encoding_name="cl100k_base"  # 指定使用 OpenAI 的分词器
)

text = "检索增强生成技术结合了信息检索与大语言模型，能够有效减少模型产生幻觉的问题。"
chunks = splitter.split_text(text)

print(f"✅ 原始文本: {text}")
print(f"✅ 生成块数: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    print(f"块 {i}: {chunk}")