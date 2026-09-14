from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """<html>
<body>
<h1>智能音箱用户手册</h1>
<h2>快速入门</h2>
<p>将音箱连接电源，下载官方App进行配网。</p>
<h2>常见问题</h2>
<p>请检查Wi-Fi密码是否正确。</p>
</body>
</html>"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_text)

for i, chunk in enumerate(chunks, 1):
    print(f"块 {i}: {chunk.page_content[:60]}...")
    print(f"  元数据: {chunk.metadata}\n")
