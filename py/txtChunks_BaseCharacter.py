def chunk_by_char(text, chunk_size=100, overlap=20):
    """按字符数分块"""
    step = chunk_size - overlap
    return [text[i:i+chunk_size] for i in range(0, len(text), step)]

# 使用示例
text = "检索增强生成技术结合了信息检索与大语言模型，能够有效减少模型产生幻觉的问题。"
chunks = chunk_by_char(text, chunk_size=20, overlap=5)

for i, chunk in enumerate(chunks):
    print(f"块{i+1}: {chunk}")