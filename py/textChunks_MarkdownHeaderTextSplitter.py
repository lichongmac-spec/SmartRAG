from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """# 智能音箱用户手册

## 快速入门
将音箱连接电源，下载官方App进行配网。

## 常见问题
### 无法连接网络
请检查Wi-Fi密码是否正确。"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

for i, chunk in enumerate(chunks, 1):
    print(f"块 {i}:")
    print(f"  元数据: {chunk.metadata}")
    print(f"  内容: {chunk.page_content[:50]}...\n")
