from langchain_text_splitters import TokenTextSplitter

# 创建分块器
splitter = TokenTextSplitter(
    chunk_size=10,
    chunk_overlap=3
)

# 执行分块
text = "检索增强生成技术结合了信息检索与大语言模型..."
chunks = splitter.split_text(text)

print(f"运行结果如下：")
print(f"切分为 {len(chunks)} 个块")