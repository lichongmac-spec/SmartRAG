import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 将 PDF 转换为结构化的 Markdown 文本，保留标题层级
#    这一步是核心，它会尝试识别文档的标题、段落等结构
md_text = pymupdf4llm.to_markdown("README.pdf")

# 2. 使用递归字符分割器，利用 Markdown 的标题（#、##）作为主要分隔符
#    这样可以在“语义完整”和“块大小”之间取得平衡
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,              # 块的最大字符数
    chunk_overlap=200,            # 块之间的重叠字符数
    separators=["\n\n", "\n", "# ", "## ", " ", ""] # 优先按段落、标题切分
)

# 3. 执行分块（假设 md_text 已被封装成 LangChain 的 Document 对象）
#    实际使用时，你可能需要先将 md_text 转为 Document，或者直接 split_text
chunks = text_splitter.split_text(md_text)

# 打印结果
for i, chunk in enumerate(chunks):
    print(f"块 {i+1}:\n{chunk[:100]}...\n")
    i=+2