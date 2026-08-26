from docx import Document

def extract_structured_chunks(docx_path):
    """从docx文档中提取带有结构信息的块（标题+内容）。"""
    doc = Document(docx_path)
    chunks = []
    current_heading = "根标题"
    current_content = []

    for para in doc.paragraphs:
        # ✅ 增加安全判断：确保 para.style 不为 None
        if para.style is not None and para.style.name and para.style.name.startswith('Heading'):
            # 如果之前有积累的内容，先保存为一个块
            if current_content:
                chunks.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content)
                })
                current_content = []
            # 更新当前标题
            current_heading = para.text.strip()
        else:
            # 非标题段落，视为正文内容（跳过空段落）
            if para.text.strip():
                current_content.append(para.text.strip())

    # 保存最后一个块
    if current_content:
        chunks.append({
            "heading": current_heading,
            "content": "\n".join(current_content)
        })
    return chunks

# --- 使用示例 ---
chunks = extract_structured_chunks("docx.docx")

for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(f"所属标题: {chunk['heading']}")
    print(f"内容预览: {chunk['content'][:100]}...\n")