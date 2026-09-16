from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 模拟原始文档
doc = Document(page_content="""第二章：RAG核心步骤。
文档加载是第一步，将PDF、Word等格式转为文本。
文本切分将长文档拆分为语义完整的块。
向量化使用嵌入模型将文本转为数值向量。
检索生成根据用户问题找到相关块并交给LLM生成答案。""")

# 父块分块器：较大的块，用于生成
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
parent_chunks = parent_splitter.split_documents([doc])

# 子块分块器：较小的块，用于检索
child_splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=5)

# 建立父子映射
parent_child_map = []
for parent in parent_chunks:
    children = child_splitter.split_text(parent.page_content)
    for child in children:
        parent_child_map.append({
            "child": child,
            "parent": parent.page_content
        })

# 打印父子映射关系
print(f"✅ 父块数量: {len(parent_chunks)}")
print(f"✅ 子块数量: {len(parent_child_map)}\n")

for i, item in enumerate(parent_child_map, 1):
    print(f"子块 {i}: {item['child']}")
    print(f"  └─ 所属父块: {item['parent'][:50]}...\n")

# 模拟检索：当子块被命中时，返回对应的父块
query_keyword = "向量化"
for item in parent_child_map:
    if query_keyword in item["child"]:
        print(f"🔍 查询 '{query_keyword}' 命中了子块:")
        print(f"   子块内容: {item['child']}")
        print(f"   ✅ 返回的父块上下文: {item['parent']}")
        break
