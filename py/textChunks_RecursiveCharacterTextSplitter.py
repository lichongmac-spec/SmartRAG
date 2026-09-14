from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """检索增强生成技术结合了信息检索与大语言模型。
它能够有效减少模型产生幻觉的问题。RAG系统的核心步骤包括文档加载、文本切分、向量化和检索生成。"""

# 默认分隔符列表：["\n\n", "\n", " ", ""]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=5
)

chunks = splitter.split_text(text)

print(f"✅ 原始文本长度: {len(text)} 字符")
print(f"✅ 生成块数: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    print(f"块 {i} (长度: {len(chunk)} 字符): {chunk}")
