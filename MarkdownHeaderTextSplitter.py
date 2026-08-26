from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. 定义要切分的标题层级，并给它一个友好的名字用于元数据
headers_to_split_on = [
    ("#", "一级标题"),
    ("##", "二级标题"),
]

# 2. 创建分割器实例
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# 3. 你的Markdown文档
markdown_document = """
# 智能音箱用户手册

## 快速入门
将音箱连接电源，下载官方App进行配网。

## 常见问题
### 无法连接网络
请检查Wi-Fi密码是否正确。
"""

# 4. 执行分割
chunks = markdown_splitter.split_text(markdown_document)

# 5. 查看结果
for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(f"元数据: {chunk.metadata}")
    print(f"内容: {chunk.page_content}\n")