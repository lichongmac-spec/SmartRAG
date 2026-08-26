import fitz  # PyMuPDF

def extract_structured_text(pdf_path):
    """从PDF中按页面提取文本块及其坐标信息，用于识别结构。"""
    doc = fitz.open(pdf_path)
    structured_data = []
    for page_num, page in enumerate(doc):
        # 获取页面上的所有文本块（包含坐标、字体大小等信息）
        blocks = page.get_text("dict")["blocks"]
        page_text = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    # 根据字号、加粗等特征可以判断是否为标题
                    # 这里简单地将所有内容按顺序收集
                    page_text.append({
                        "text": span["text"],
                        "font_size": span["size"],
                        "font": span["font"],
                        "bbox": span["bbox"]
                    })
        structured_data.append({"page": page_num, "content": page_text})
    return structured_data

# 使用示例
data = extract_structured_text("README.pdf")
print(f"提取到 {len(data)} 页的结构化数据，之后可以基于 'font_size' 等字段来划分结构。")