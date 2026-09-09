import tiktoken

# 初始化 OpenAI 的 cl100k_base 分词器
encoding = tiktoken.get_encoding("cl100k_base")

text = "检索增强生成技术结合了信息检索与大语言模型。"
tokens = encoding.encode(text)

print(f"原始文本: {text}")
print(f"文本字符数: {len(text)}")
print(f"文本词元数: {len(tokens)}")
print(f"词元序列 (前10个): {tokens[:10]}")