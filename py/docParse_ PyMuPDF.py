import fitz  # PyMuPDF
import os

# 打印当前工作目录，确认执行位置
print(f"当前工作目录: {os.getcwd()}")
# 打开PDF
doc = fitz.open("README.pdf")
print(f"✅ 成功打开PDF，共 {len(doc)} 页\n")

# 遍历提取每页文本
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    print(f"--- Page {page_num + 1} ---")
    print(f"本页字符数: {len(text)}")
    print(f"内容预览: {text[:100]}...\n")  # 打印前200字符

doc.close()
print("✅ 处理完成，文档已关闭")