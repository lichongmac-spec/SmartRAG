from langchain_text_splitters import PythonCodeTextSplitter  # 导入Python代码专用分块器

python_code = """
class Foo:
    def bar(self):
        pass

def foo():
    pass
"""

# 创建分块器：每块最多30字符，无重叠
splitter = PythonCodeTextSplitter(chunk_size=30, chunk_overlap=0)

# 按Python语法（class/def）切分代码
chunks = splitter.split_text(python_code)

# 打印每个代码块
for i, chunk in enumerate(chunks, 1):
    print(f"块 {i}: {repr(chunk)}")