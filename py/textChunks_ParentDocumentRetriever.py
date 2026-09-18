from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# 旧导入（已失效）
# from langchain_core.storage import InMemoryStore

# 新导入（正确）
# from langgraph.store.memory import InMemoryStore
from langchain_core.stores import InMemoryStore          # ✅ 改为 langchain_core.stores
# from langchain.retrievers import ParentDocumentRetriever
# from langchain_text_splitters.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

# 1. 准备文档
docs = [
    Document(page_content="""检索增强生成（RAG）的核心步骤包括：
文档加载：将PDF、Word等格式转为文本。
文本切分：将长文档拆分为语义完整的块。
向量化：使用嵌入模型将文本转为数值向量。
检索生成：根据用户问题找到相关块并交给LLM。""")
]

# 2. 创建父子分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)  # 大块
child_splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=5)    # 小块

# 3. 初始化向量存储和文档存储
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents([Document(page_content="placeholder")], embeddings)
store = InMemoryStore()

# 4. 创建父子块检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 5. 添加文档（自动完成父子分块和索引）
retriever.add_documents(docs)

# 6. 执行检索
query = "向量化是怎么做的？"
retrieved_docs = retriever.invoke(query)

print(f"🔍 查询: {query}")
print(f"✅ 检索到 {len(retrieved_docs)} 个父块\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"--- 父块 {i} ---")
    print(doc.page_content)
