# 1. 从 langchain_community 导入 PDF 加载器
from langchain_community.document_loaders import PyPDFLoader
# 2. 创建加载器实例，指定 PDF 文件路径
loader = PyPDFLoader("README.pdf")  # 请替换为你的文件路径
# 3. 加载文档：将 PDF 内容读取为 LangChain 的 Document 对象列表
documents = loader.load()

# 4. （可选）查看文档数量，验证加载结果
print(f"成功加载 {len(documents)} 页内容。")

# 5. （可选）打印第一页的部分内容，查看提取的文本
print(documents[0].page_content[:100])  # 打印前100个字符