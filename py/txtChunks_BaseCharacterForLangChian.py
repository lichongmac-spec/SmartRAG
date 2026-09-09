from langchain.text_splitter import CharacterTextSplitter

text = "检索增强生成技术结合了信息检索与大语言模型，能够有效减少模型产生幻觉的问题。"

splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
chunks = splitter.split_text(text)

# 打印验证
print(f"✅ 原始文本长度: {len(text)} 字符")
print(f"✅ 生成块数: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    print(f"块 {i} (长度: {len(chunk)} 字符): {chunk}")