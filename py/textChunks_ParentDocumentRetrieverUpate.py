# ==================== 导入区（全部使用最新推荐路径） ====================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma                           # 替代 FAISS，无社区包日落警告
from langchain_huggingface import HuggingFaceEmbeddings       # 替代 langchain_community 版本
from langchain_core.stores import InMemoryStore               # 正确类型，匹配 ParentDocumentRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

# ==================== 1. 准备文档 ====================
docs = [
    Document(page_content="""检索增强生成（RAG）的核心步骤包括：
文档加载：将PDF、Word等格式转为文本。
文本切分：将长文档拆分为语义完整的块。
向量化：使用嵌入模型将文本转为数值向量。
检索生成：根据用户问题找到相关块并交给LLM。""")
]

# ==================== 2. 创建父子分割器 ====================
# 父块：增大 chunk_size，确保一个完整小节不被拆散；加入中文标点作为分隔符
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,          # 父块最大字符数，足以容纳整段说明
    chunk_overlap=40,        # 20% 重叠
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

# 子块：较小，用于精准检索
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,           # 子块最大字符数
    chunk_overlap=10,        # 20% 重叠
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

# ==================== 3. 初始化嵌入模型与向量存储 ====================
# 中文优化的嵌入模型（若需英文可换回 all-MiniLM-L6-v2）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True}   # 归一化提升余弦相似度准确性
)

# 使用 Chroma 作为向量存储（内存模式，适合演示）
vectorstore = Chroma(
    collection_name="parent_child_demo",
    embedding_function=embeddings
)

# 文档存储：保存父块原始内容
store = InMemoryStore()

# ==================== 4. 创建父子块检索器 ====================
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 3}        # 返回 Top-3 父块
)

# ==================== 5. 添加文档（自动完成父子分块和索引） ====================
retriever.add_documents(docs)

# ==================== 6. 执行检索 ====================
query = "向量化是怎么做的？"
retrieved_docs = retriever.invoke(query)

print(f"🔍 查询: {query}")
print(f"✅ 检索到 {len(retrieved_docs)} 个父块\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"--- 父块 {i} ---")
    print(doc.page_content)
    print()

# 🔍 查询: 向量化是怎么做的？
# ✅ 检索到 1 个父块

# --- 父块 1 ---
# 检索增强生成（RAG）的核心步骤包括：
# 文档加载：将PDF、Word等格式转为文本。
# 文本切分：将长文档拆分为语义完整的块。
# 向量化：使用嵌入模型将文本转为数值向量。
# 检索生成：根据用户问题找到相关块并交给LLM。