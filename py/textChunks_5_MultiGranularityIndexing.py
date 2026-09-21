# 多粒度索引：三级索引 + 动态粒度选择 + 关键词匹配检索
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. 文档：两个章节
doc = Document(page_content="""第二章：RAG核心步骤。
文档加载是第一步，将PDF、Word等格式转为文本。
文本切分将长文档拆分为语义完整的块。
向量化使用嵌入模型将文本转为数值向量。
检索生成根据用户问题找到相关块并交给LLM生成答案。

第三章：高级检索技术。
混合检索结合向量检索和关键词检索。
重排序使用交叉编码器对候选文档重新打分。""")

# 2. 章节级：按标题切分，确保每章独立
chapter_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150, chunk_overlap=0,
    separators=["\n\n第三章", "\n\n", "\n", "。", " ", ""]
)
# 句子/段落级
sentence_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)
paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)

sentences = sentence_splitter.split_text(doc.page_content)
paragraphs = paragraph_splitter.split_text(doc.page_content)
chapters = chapter_splitter.split_text(doc.page_content)

print(f"✅ 句子级: {len(sentences)} 块")
print(f"✅ 段落级: {len(paragraphs)} 块")
print(f"✅ 章节级: {len(chapters)} 块\n")

# 3. 三级映射
sentence_to_paragraph = []
for s in sentences:
    for p in paragraphs:
        if s in p:
            sentence_to_paragraph.append({"sentence": s, "paragraph": p})
            break

paragraph_to_chapter = []
for p in paragraphs:
    for c in chapters:
        if p in c:
            paragraph_to_chapter.append({"paragraph": p, "chapter": c})
            break

# 4. 关键词提取：按连续中文/字母数字切分，过滤停用词
STOPWORDS = {"的", "是", "有", "哪些", "什么", "怎么", "如何", "核心", "步骤", "？", "?"}

def extract_keywords(query):
    # 按连续中文/字母数字切分，保留长度>=2的词
    tokens = re.findall(r'[\u4e00-\u9fa5]+|[A-Za-z0-9]+', query)
    keywords = []
    for t in tokens:
        if t in STOPWORDS or len(t) < 2:
            continue
        keywords.append(t)
        # 中文长词再切2-3字子串，增加匹配粒度
        if len(t) > 3 and re.match(r'[\u4e00-\u9fa5]+', t):
            for length in [3, 2]:
                for i in range(len(t) - length + 1):
                    sub = t[i:i+length]
                    if sub not in STOPWORDS:
                        keywords.append(sub)
    # 去重，按长度降序
    return sorted(set(keywords), key=len, reverse=True)

# 5. 多粒度检索：动态选择粒度 + 关键词匹配
def retrieve(query, granularity="auto"):
    keywords = extract_keywords(query)
    print(f"   提取关键词: {keywords}")
    
    if granularity == "auto":
        granularity = "sentence" if len(query) <= 8 else "paragraph"
    
    if granularity == "sentence":
        for item in sentence_to_paragraph:
            if any(kw in item["sentence"] for kw in keywords):
                return {"hit_level": "句子级", "hit_block": item["sentence"],
                        "context_level": "段落级", "context_block": item["paragraph"]}
    elif granularity == "paragraph":
        for item in paragraph_to_chapter:
            if any(kw in item["paragraph"] for kw in keywords):
                return {"hit_level": "段落级", "hit_block": item["paragraph"],
                        "context_level": "章节级", "context_block": item["chapter"]}
    return None

# 6. 演示两个查询
queries = ["向量化", "RAG的核心步骤有哪些？"]
for q in queries:
    print(f"🔍 查询: {q}")
    result = retrieve(q, granularity="auto")
    if result:
        print(f"   命中粒度: {result['hit_level']}")
        print(f"   命中块: {result['hit_block'][:50]}...")
        print(f"   返回上下文({result['context_level']}): {result['context_block'][:60]}...\n")
    else:
        print("   未命中\n")