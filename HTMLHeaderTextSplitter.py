from langchain_text_splitters import HTMLHeaderTextSplitter

# 1. 定义按哪些HTML标签来分割
headers_to_split_on = [
    ("h1", "主标题"),
    ("h2", "次标题"),
]

# 2. 创建分割器实例
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 3. 你的HTML文档
html_document = """
<html>
<body>
    <h1>智能音箱用户手册</h1>
    <h2>快速入门</h2>
    <p>将音箱连接电源，下载官方App进行配网。</p>
    <h2>常见问题</h2>
    <p>请检查Wi-Fi密码是否正确。</p>
</body>
</html>
"""

# 4. 执行分割
chunks = html_splitter.split_text(html_document)

# 5. 查看结果
for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(f"元数据: {chunk.metadata}")
    print(f"内容: {chunk.page_content}\n")