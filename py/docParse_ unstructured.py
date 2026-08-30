from unstructured.partition.auto import partition
import os

# 检查文件是否存在
doc_path = "example.docx"
if not os.path.exists(doc_path):
    print(f"❌ 文件不存在: {doc_path}")
    exit()

# 自动识别并解析文件
elements = partition(filename=doc_path)
print(f"✅ 解析成功，共提取 {len(elements)} 个元素\n")

# 打印所有元素
for i, element in enumerate(elements):
    print(f"--- 元素 {i+1} ---")
    print(f"类型: {element.category}")
    print(f"内容预览: {str(element.text)[:150]}...\n")