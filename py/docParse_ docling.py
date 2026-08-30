from docling.document_converter import DocumentConverter
import os

# 检查文件是否存在
pdf_path = "README.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ 文件不存在: {pdf_path}")
    exit()

# 创建转换器并转换
converter = DocumentConverter()
result = converter.convert(pdf_path)

# 导出为Markdown
markdown_output = result.document.export_to_markdown()
print(f"✅ 转换成功，共 {len(markdown_output)} 字符")
print(f"--- 预览 (前200字符) ---")
print(markdown_output[:200])