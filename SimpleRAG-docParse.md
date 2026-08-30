# 文档解析


常用文档格式:

* JSON
* Docx
* PDF:解析工具有：PyPDFLoader｜PyPDFium2|PDFMiner|PyMuPDF
* CSV
* MarkDown
* HTML
* TXT
* XML



文档解析是构建RAG系统的第一步，也是决定整个系统效果的上限。它的核心任务是将非结构化的原始文档（如PDF、Word、扫描件、图片）转化为结构化的、机器可读的数据，为后续的文本切分、向量化和检索奠定基础。

### 📝 为什么文档解析如此重要？

可以这样理解：如果向量数据库是RAG系统的“仓库”，文档解析就是“进货质检员”。如果这一步提取的信息是残缺或错误的，后续无论检索模型多优秀、大模型多强大，都难以产生准确、可靠的回答。

具体来说，文档解析需要解决以下关键挑战：

1.  **提取文本内容**：将文档中的文字准确抓取出来，不只是纯文本，更要理解其**结构**（标题层级、段落、列表）、**表格**和**图片**内容。
2.  **处理复杂版面**：识别多栏布局、页眉页脚、脚注等，确保内容提取的**阅读顺序**正确。
3.  **应对扫描件与图像**：对于扫描版PDF或图片，需要使用**OCR（光学字符识别）** 技术识别其中的文字。

### 🛠️ 主流文档解析方案对比

目前没有一种方案是万能的，需要根据你的文档类型、预算和技术栈来选择。以下是几个主流方向和工具对比：

| 方案类型 | 代表工具/服务 | 核心优势 | 主要局限 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **开源Python库** | **PyMuPDF (fitz)**、**Unstructured**、**Docling** | 免费、灵活、可控、可本地部署 | 处理复杂版面需要额外配置，部分工具（如Unstructured）对嵌套表格支持不佳 | 开发者主导、预算有限、对数据隐私要求高、文档格式相对规整的项目 |
| **商业API服务** | **Azure AI Document Intelligence**、**百度智能文档处理**、**腾讯云文档解析** | **开箱即用**、精度高（尤其复杂表格和版面）、持续更新、支持多模态 | 按量付费，长期成本可能较高；数据需上传至云端（尽管有私有化部署选项） | 追求快速上线、对解析精度要求苛刻、不想投入大量工程资源维护解析管线的团队 |
| **专业解析工具** | **TextIn xParse** | 专为RAG场景优化，输出结果对LLM友好 | 通常为商业产品，开源版本功能可能受限 | 需要高质量解析结果，特别是处理复杂文档（如财务报告、科研论文）的RAG应用 |

### 🚀 如何选择适合你的方案？

你可以根据以下思路快速决策：

1.  **文档类型是“数字原生”（如Word/PPT另存的PDF）还是“扫描件”？**
    *   **数字原生PDF**：优先考虑**PyMuPDF**，它速度极快且提取效果不错。如果想进一步保留结构和表格，**Docling**或**Unstructured**是更好的选择。
    *   **扫描件/图片PDF**：必须使用带有**OCR功能**的方案。**Docling**和各大云服务商（Azure、百度、腾讯云）的文档智能服务都内置了优秀的OCR引擎。

2.  **文档中表格和版面是否复杂？**
    *   **简单表格**：大部分工具都能胜任。
    *   **嵌套表格、合并单元格、复杂排版**：**商业API服务**的精度优势会非常明显。一些开源工具（如Watson Discovery和Unstructured）在处理这类复杂文档时表现不佳。

3.  **项目对数据隐私和合规性要求如何？**
    *   **高**：务必选择**本地部署**的开源方案，如**PyMuPDF**、**Docling**等。
    *   **低**：可以放心使用便捷的**云API服务**，可以大幅提升开发效率。

### 💡 额外的建议

*   **格式转换技巧**：如果某工具对Word (.docx) 解析效果不佳，可考虑将其转换为PDF后再解析，有时能获得更好的表格提取效果。
*   **从小处着手**：建议先用几种不同的工具对一小批有代表性的文档进行测试，对比输出结果，然后再决定最终方案。


## 开源Python库PyMuPDF、Docling、Unstructured

简单来说，**PyMuPDF** 是追求极致速度的“性能之王”，**Docling** 是擅长处理复杂表格和布局的“高精度专家”，而 **Unstructured** 则是为生产级RAG系统设计的“全能平台”。

我把它们的特点和适用场景整理成了下面这个表格，方便你对比：

| 特性 | PyMuPDF (fitz) | Unstructured | Docling |
| :--- | :--- | :--- | :--- |
| **一句话定位** | PDF处理的“瑞士军刀”，以速度和全面性著称 | RAG数据摄入的“全能平台”，支持海量格式 | 文档转Markdown的“高精度专家”，擅长表格与布局 |
| **核心优势** | **极速**：底层是C语言，处理速度可快8-15倍<br>**功能全面**：可提取文本、图片、元数据，甚至编辑PDF | **格式支持最广**：支持超过65种文件格式<br>**企业级功能**：提供SaaS、API和本地部署选项 | **AI驱动**：内置布局分析、TableFormer和OCR模型<br>**高精度表格还原**：对复杂表格结构恢复能力突出 |
| **主要局限** | **无法处理扫描件**：对纯图片PDF无能为力<br>**表格提取较原始**：需手动处理坐标数据 | **处理复杂表格有短板**：在处理嵌套表格时可能不完美<br>**商业版按量计费**：大规模使用成本可能较高 | **首次使用“重”**：需下载约500MB的AI模型，首次启动慢<br>**处理速度较慢**：约为0.3-1秒/页 |
| **适用场景** | **数字生成PDF**的快速文本提取<br>**批量处理**大量简单文档<br>需要**原生集成到Python脚本**中 | **企业级RAG流水线**<br>处理**格式混杂**的大量文档<br>需要**多种部署方案**（云/SaaS/本地） | 对**表格和版面还原精度**要求极高（如财报、科研论文）<br>需要**高质量Markdown输出**的RAG应用 |

### 📝 简单上手示例

#### 1. PyMuPDF (fitz)：极速文本提取
它非常适合快速抓取数字PDF中的文字。

```python
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

```
#### 运行输出示例

```
/github/SmartRAG/py/docParse_\ PyMuPDF.py 
当前工作目录: ../Documents/github/SmartRAG
✅ 成功打开PDF，共 20 页

--- Page 1 ---
本页字符数: 1648
内容预览: RAG(Retrieval-Augmented Generation，检索增强⽣成)
5 步理解核⼼流程
# ==================== 第 1 步：准备知识库（3 条⽂档） =====...

--- Page 2 ---
本页字符数: 1369
内容预览:  检索到的相关⽂档：
  1. RAG 的核⼼步骤：加载⽂档 → 切分 → 向量化 → 检索 → ⽣成。
  2. RAG 是检索增强⽣成，它结合了信息检索和⼤语⾔模型。
 ⽣成的答案：
根据知识库，...

....

✅ 处理完成，文档已关闭

```

#### 2. Docling：精准转换复杂文档(当前电脑环境不满足运行要求)
它专注于将文档高精度地转换为Markdown或JSON格式。
##### 安装

```
pip install docling
```

代码：


```python
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
```

### 运行输出示例
```
✅ 转换成功，共 2847 字符
--- 预览 (前200字符) ---
# 检索增强生成（RAG）技术指南

RAG（Retrieval-Augmented Generation）是一种结合信息检索与大语言模型的技术框架。

本指南将系统介绍RAG的核心概念、实现步骤及最佳实践。

## 第一章：RAG概述

RAG...
```

### 代码关键点

*   **`DocumentConverter()`**：创建转换器实例，自动处理文档格式
*   **`converter.convert()`**：执行转换，返回转换结果对象
*   **`result.document`**：获取解析后的文档对象
*   **`export_to_markdown()`**：导出为Markdown格式文本
*   **`export_to_json()`**：可替代方法，导出为JSON格式

### 输出格式切换

```python
# 输出为JSON
json_output = result.document.export_to_json()
print(json_output)

# 输出为纯文本
text_output = result.document.export_to_text()
print(text_output)
```

### 注意事项

*  **首次运行**：Docling 会自动下载约 **500MB** 的 AI 模型文件，需要联网，耐心等待。
*  **支持格式**：PDF、DOCX、HTML、CSV、Markdown 等常见格式。
*  **耗时**：处理速度约 0.3-1 秒/页，复杂表格和图片会增加耗时。

### 更多扩展
Docling 的安装和使用都比较直接，主要分为基础安装和可能遇到的几个小问题。

#### 🚀 安装步骤

最推荐的方式就是直接用 pip 安装完整版，它能开箱即用地处理大部分文档。

```bash
pip install docling
```

#### 几个小建议

根据其他开发者的经验，有几点可以注意一下：

1.  **隔离环境**：为了避免和已有项目的依赖冲突，建议创建一个新的虚拟环境（比如用 `conda` 或 `python -m venv`）。项目里用的是 **venv**
2.  **依赖版本**：如果在安装或运行时遇到和 `click` 或 `typer` 相关的错误，可以尝试先安装特定版本：`pip install typer==0.9.0 click==8.1.7`，再装 Docling。
3.  **首次运行**：第一次使用 Docling 时，它会自动下载一些 AI 模型（大约 500MB），需要联网并等待一会儿。

#### 💻 验证安装与基本使用

你可以用最简单的代码来验证是否安装成功，并感受一下它的用法。

1.  **测试 Python API**：创建一个 Python 文件，写入以下代码并运行。


    ```python
    from docling.document_converter import DocumentConverter

    # 可以处理本地文件或 URL
    source = "https://arxiv.org/pdf/2408.09869"  # 或者替换为你的本地PDF路径
    converter = DocumentConverter()
    result = converter.convert(source)

    # 输出 Markdown 格式内容到控制台
    print(result.document.export_to_markdown())
    ```

2.  **测试命令行工具**：在终端中直接运行，可以快速将文档转为 Markdown。


    ```bash
    docling https://arxiv.org/pdf/2206.01062
    # 或者处理本地文件
    docling your_document.pdf
    ```
    如果看到帮助信息或开始处理文档，就说明安装没问题。

#### 🖥️ 高级选项：Docling Serve

如果你想通过 Web 界面或 API 来使用 Docling，官方提供了 `Docling Serve` 的 Docker 镜像，可以很方便地部署成一个服务。

```bash
# 拉取镜像并运行容器
docker run -d \
  --name docling-serve \
  -p 5300:5001 \
  -e DOCLING_SERVE_ENABLE_UI=1 \
  quay.io/docling-project/docling-serve:v1.9.0
```
运行后，通过浏览器访问 `http://localhost:5300` 就能看到操作界面了。如果是团队作业，建立一个 docling Serve 很有用

#### 3. Unstructured：处理混合格式文档（当前电脑环境不满足运行要求）
它以“元素”为单位解析文档，便于后续细粒度处理。

```python
from unstructured.partition.auto import partition

# 自动识别并解析文件
elements = partition(filename="example.docx")

# 打印文档中的每个“元素”（如标题、段落、表格）
for element in elements:
    print(element)
```

## 常用格式处理

这三款工具对常见文档格式的支持，确实各有侧重。我把它们对你列出的格式的支持情况和用法整理了一下，方便你对比。

简单来说：处理纯文本和 PDF，它们都擅长，但方向不同；办公文档（docx, xlsx）和富文本（HTML, XML）需要看它们各自的支持深度；而像 JSON、CSV 这种结构化数据，它们更多是作为**输出**或**解析目标**来处理的。

### 📄 各工具对常见格式的处理方式

| 格式 | PyMuPDF (fitz) | Unstructured | Docling |
| :--- | :--- | :--- | :--- |
| **PDF** | ✅ **原生支持**。是其核心功能。 | ✅ **原生支持**。可通过 `hi_res` 策略进行布局分析。 | ✅ **原生支持**。内置 AI 模型进行深度解析。 |
| **DOCX** | ✅ **支持**。通过 `PyMuPDF Pro` (商业版) 可原生打开并提取文本。 | ✅ **原生支持**。是其主要支持的办公文档格式之一。 | ✅ **原生支持**。支持 DOCX 作为输入格式。 |
| **TXT** | ✅ **支持**。可用 `filetype="txt"` 参数打开任何纯文本文件。 | ✅ **原生支持**。支持 `.txt` 文件。 | ✅ **支持**。可输出为纯文本格式。 |
| **HTML** | ⚠️ **有限支持**。可用 `filetype="txt"` 打开，但不会解析 HTML 结构，只提取文本内容。 | ✅ **原生支持**。可以解析 HTML 文件，提取其中的结构化内容。 | ✅ **原生支持**。支持 HTML 作为输入，并保留文档结构。 |
| **XML** | ⚠️ **有限支持**。用法同 HTML，仅提取文本，不解析 XML 结构。 | ✅ **原生支持**。可以解析 XML 文件，并支持保留 XML 标签选项。 | ✅ **原生支持**。支持多种特定 schema 的 XML (如 JATS, USPTO)。 |
| **Markdown** | ✅ **主要作为输出**。可用 `PyMuPDF4LLM` 将文档转换为 Markdown。 | ✅ **原生支持**。支持 `.md` 文件作为输入。 | ✅ **双向支持**。支持 Markdown 作为输入和输出格式。 |
| **JSON** | ✅ **主要作为输出**。可用 `PyMuPDF4LLM` 将文档转换为 JSON。 | ✅ **原生支持**。支持 `.json` 文件作为输入。 | ✅ **主要作为输出**。是其 "Docling Document" 无损序列化的核心输出格式。 |
| **CSV** | ❌ **无直接支持**。未找到直接解析或导出 CSV 的功能。 | ✅ **原生支持**。支持 `.csv` 文件作为输入。 | ✅ **原生支持**。支持 CSV 作为输入格式。 |

> **注意**：表格中的“原生支持”指工具提供专门的文件解析逻辑，而“有限支持”则指工具能打开文件，但只能提取原始文本，无法理解其结构。


### 💎 建议
*   **如果你主要处理 PDF 并想快速接入 RAG**：首选 **PyMuPDF** (`pymupdf4llm`)，它轻量、极速，且输出格式对 LLM 友好。
*   **如果你需要处理格式混杂的大量文档（如邮件、HTML、CSV）**：**Unstructured** 提供了最广泛的开箱即用支持，是一个稳定的企业级选择。
*   **如果你追求深度解析和统一的文档表示，尤其包含复杂表格**：**Docling** 的输入输出格式都很丰富，是理想的“文档处理中枢”。

## 💎 如何选择？
你可以遵循一个简单的决策思路：

1.  **如果你的文档都是数字生成的PDF，且只关心纯文本内容**：**PyMuPDF** 是最简单、快速的选择，它能覆盖你90%以上的需求。
2.  **如果你的文档包含大量复杂表格、扫描件，或者你追求RAG应用的最佳效果**：可以优先尝试 **Docling**。它在官方基准测试中表现优异。如果Docling因为模型太重或速度不满足需求，**Unstructured** 是一个很好的企业级替代方案。
3.  **如果你在处理一个大规模、多格式的文档库（如PDF、Word、HTML、邮件等），且需要构建企业级的RAG平台**：**Unstructured** 会是最适合的，因为它支持的文件格式最广，并提供完善的API和企业级功能。

## 商业API服务

这三个工具都是大厂提供的商业级文档解析云服务，它们可以看作是“开箱即用”的智能文档处理平台，很适合在生产环境中快速落地。我把它们的核心差异和使用场景整理了一下。

### 📊 核心能力对比

| 特性 | Azure AI Document Intelligence | 百度智能文档解析 | 腾讯云文档解析 |
| :--- | :--- | :--- | :--- |
| **底层技术** | 基于Azure的认知服务，提供预训练和自定义模型 | 基于**PaddleOCR-VL 1.6**多模态大模型及开源底座 | 基于自研**OCR解析大模型（DocLM）** 与**多模态大模型（MLLM）** |
| **核心优势** | **预训练模型丰富**：针对发票、合同、身份证等特定文档类型有大量优化模型<br>**高度定制化**：支持自定义模型训练，甚至可以用生成式AI提取非结构化数据 | **格式支持广**：支持doc、pdf、图片、xlsx等**18种**主流格式<br>**输出友好**：直接返回**Markdown/JSON**结构化内容<br>**内置切分**：可按标题层级切分长文档 | **复杂元素解析强**：对**复杂数学公式**、**数据图表**（Chart2Table）、**逻辑拓扑图**等解析专项优化<br>**智能语义切分**：具有业内首个**语义切分大模型**，避免截断语义 |
| **适用场景** | 企业级自动化流程（应付账款、保险、银行表单处理），RPA集成 | 通用文档结构化（政务、企业知识库），尤其适合中文文档和多种格式混合场景 | 科研论文（公式多）、金融研报（图表多）、说明书（版式复杂）等高精度解析场景 |
| **主要局限** | **中文支持有历史短板**：据早期评测，对中国复杂表格和无框表格的识别可能不如国内厂商 | **复杂表格可能一般**：早期评测中，对**无表格线**的复杂表格识别表现不稳定 | **资源较重**：依赖大模型，服务响应延迟和成本可能比传统OCR方案高 |

### 📝 简单的上手示例

这三家都是标准的RESTful API服务，调用方式很相似。

#### Azure AI Document Intelligence (Python SDK)

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

endpoint = "你的endpoint"
key = "你的key"
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

with open("你的文档.pdf", "rb") as f:
    poller = client.begin_analyze_document("prebuilt-layout", document=f)
result = poller.result()

# 提取所有段落和表格
for paragraph in result.paragraphs:
    print(paragraph.content)
for table in result.tables:
    for cell in table.cells:
        print(cell.content)
```

#### 百度智能文档解析 (API示例)

```python
import requests

url = "https://aip.baidubce.com/rest/2.0/ocr/v1/doc_parser"
params = {"access_token": "你的AccessToken"}
headers = {"Content-Type": "application/x-www-form-urlencoded"}

with open("你的文档.pdf", "rb") as f:
    data = {"file": f.read()}
    response = requests.post(url, params=params, headers=headers, data=data)
    result = response.json()
    
# 输出解析后的Markdown内容
print(result["data"]["markdown"])
```

#### 腾讯云文档解析 (API示例)

```python
import requests

# 腾讯云文档解析API
url = "https://lke.tencentcloudapi.com/"
params = {
    "Action": "ParseDoc",
    "Version": "2023-11-30",
    "DocUrl": "你的文档URL或Base64"
}
headers = {"Authorization": "你的签名"}

response = requests.post(url, json=params, headers=headers)
result = response.json()
print(result["Response"]["MarkdownContent"])
```

> 注意：以上只是简化的调用示意，实际使用需要参考各平台最新的API文档获取签名、认证方式及具体的请求/响应格式。

### 💡 选型建议

根据自己的实际情况来选就好：

*   **如果你的业务有大量国际化需求，或需要与微软生态（如Power Automate）深度集成**：**Azure AI Document Intelligence** 是一个功能全面、值得信赖的企业级选择。
*   **如果你处理的是通用文档，希望中文识别效果好、支持格式多、集成简单**：**百度智能文档解析** 的性价比和便捷性都很不错。
*   **如果你的文档以科研论文、金融研报为主，包含大量复杂公式、图表，对解析精度要求非常苛刻**：可以优先考虑**腾讯云文档解析**，它为这些场景做了专门的优化。

### 处理日常文档介绍
这三家商业文档解析服务都面向企业级应用，对常见格式的支持已非常完善，但在**输入格式的广度**和**输出内容的深度**上各有侧重。

### 📄 格式支持与输出方式总览

| 格式 | Azure AI Document Intelligence | 百度智能文档处理 | 腾讯云文档解析 |
| :--- | :--- | :--- | :--- |
| **PDF** | ✅ 原生支持 | ✅ 原生支持 | ✅ 原生支持 |
| **DOCX** | ✅ 支持 (需特定API版本)  | ✅ 原生支持  | ✅ 原生支持  |
| **TXT** | ⚠️ 间接支持 (作为Office文件限制) | ✅ 原生支持  | ✅ 原生支持  |
| **HTML** | ✅ 原生支持  | ✅ 原生支持  | ✅ 原生支持  |
| **JSON** | ❌ 主要作为**输出**格式  | ❌ 主要作为**输出**格式  | ❌ 主要作为**输出**格式  |
| **Markdown** | ❌ 主要作为**输出**格式  | ✅ 支持输入，并作为核心**输出**格式  | ✅ 支持输入，并作为核心**输出**格式  |
| **CSV** | ❌ 未明确支持 | ❌ 未明确支持 (XLSX可间接转换) | ✅ 支持输入  |
| **XML** | ❌ 未明确支持 | ❌ 未明确支持 | ❌ 未明确支持 |

> **说明**：三家服务均**不支持将XML作为输入格式**，您需要先将XML文件转换为支持的格式（如HTML、PDF等）后再进行解析。

### 📝 使用要点与代码示例

#### Azure AI Document Intelligence
**特点**：与微软生态深度集成，需要留意不同模型(API版本)对DOCX、XLSX等格式的支持情况。

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

endpoint = "您的Endpoint"
key = "您的Key"
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# 使用prebuilt-layout模型处理多种格式，返回结构化JSON
with open("您的文档.docx", "rb") as f:
    poller = client.begin_analyze_document("prebuilt-layout", document=f)
result = poller.result()

# result即包含所有解析内容的JSON对象
print(result) 
```

#### 百度智能文档处理
**特点**：格式支持列表广泛，明确返回**Markdown和JSON**格式结果，对中文文档优化好。

```python
import requests

url = "https://aip.baidubce.com/rest/2.0/ocr/v1/doc_parser"
params = {"access_token": "您的AccessToken"}

with open("您的文档.pdf", "rb") as f:
    response = requests.post(url, params=params, data={"file": f.read()})
    result = response.json()

# 输出Markdown格式内容
print(result["data"]["markdown"]) 
# 或输出完整JSON结构
print(result["data"]["json_result"]) 
```

#### 腾讯云文档解析
**特点**：支持的输入格式**最为广泛**，明确涵盖**CSV、MD、TXT**，并将内容智能转换为**Markdown/JSON/HTML**格式。

```python
# 需使用腾讯云官方SDK或签名调用API，以下为逻辑示例
import requests

# 根据腾讯云API文档生成签名
# ...
response = requests.post("https://lke.tencentcloudapi.com/", json={
    "Action": "ParseDoc",
    "FileType": "CSV", # 支持PDF, DOCX, CSV, MD, TXT, HTML等
    "FileUrl": "您的文件URL" # 或FileBase64
})
result = response.json()
print(result["Response"]["MarkdownContent"]) # 输出Markdown结果
```

### 💎 总结与建议
*   **格式最全**：如果处理**CSV、Markdown等**小众格式较多，**腾讯云文档解析**支持最全面。
*   **输出灵活**：如果下游任务需要**Markdown**格式，**百度**和**腾讯**支持较好；而**Azure**主要以结构化JSON输出为主，适合进一步编程处理。
*   **微软生态**：如果你的技术栈本身以微软产品为主，**Azure**集成体验会最流畅。

如果你是初次接入，建议优先选择**腾讯云**或**百度**，它们的文档和示例通常更贴合中文开发者的使用习惯。


## 专业解析工具

TextIn xParse 是合合信息 (TextIn) 推出的一款**商业级通用文档解析服务**，专为大模型、RAG（检索增强生成）系统和 AI Agent 设计。它的目标是把散乱、非结构化的原始文档（如PDF、Word、图片等）高效地转化为大模型容易“理解”和使用的结构化数据（如Markdown、JSON格式）。

### 核心特点与优势

1.  **高精度结构化与多模态元素解析**：xParse不只是识别文字，还能深度还原文档的“骨架”。它能精准识别文本、表格（包括复杂的跨页和无线表格）、公式、图像、页眉页脚、公章、手写体等超过16种内容元素，并还原它们的层级关系和阅读顺序。在针对金融年报的测试中，经xParse解析后搭建的RAG问答准确率超过99%，远高于传统OCR方案的32%。

2.  **极快的解析速度**：处理一份百页级PDF文档，解析时间最快约为**1.5到2秒**。

3.  **全格式支持与标准化输出**：支持PDF（包括扫描件和加密PDF）、Word、Excel、PPT、图片等十余种格式，解析结果可直接导出为**Markdown或JSON**等格式，方便下游任务直接使用。

4.  **灵活的集成与部署方式**：提供**实时API、离线SDK和私有化部署**等多种方案，已在**OpenClaw、Dify、WorkBuddy**等主流Agent生态中提供快捷集成。

### 简单的使用示例

xParse的使用方式非常灵活，最常用的是通过API或现成的Skill集成。

*   **方式一：在Agent平台中一句话触发（以OpenClaw为例）**
    安装xParse Skill后，只需用自然语言下达指令即可，无需编写代码。例如，在对话框中直接说：“帮我读一下这份PDF合同，提取关键条款”，Agent就会自动调用xParse完成解析并执行后续任务。

*   **方式二：直接调用API**
    对于开发者，最直接的方式是通过调用它的同步API，将解析结果保存为Markdown或JSON文件。
    
    ```python
    import requests

    # 准备请求
    url = "https://api.textin.com/parse"  # 以实际API地址为准
    headers = {"x-ti-app-id": "你的AppId", "x-ti-secret-code": "你的SecretCode"}
    # 可以设置解析模式、密码等参数，支持PDF、Word、图片等多种格式
    params = {
        "parse_mode": "auto",  # 自动选择最佳解析模式
        "apply_document_tree": 1, # 生成Markdown标题层级
        "table_flavor": "html", # 输出HTML格式表格，或选择"md"
        "formula_level": 0,  # 识别公式
    }

    # 上传你的PDF文件进行解析
    with open("your_document.pdf", "rb") as f:
        response = requests.post(url, params=params, headers=headers, data=f)

    if response.status_code == 200:
        result = response.json()
        # 从返回结果中提取Markdown格式的内容并保存
        markdown_content = result["data"]["markdown"]
        with open("output.md", "w") as f:
            f.write(markdown_content)
        print("解析完成！")
    else:
        print(f"解析失败，错误代码：{response.status_code}")
    ```

> **注意**：以上代码仅为演示核心逻辑，实际使用时需替换为最新的API地址和你的认证凭证，并参考官方文档处理完整的请求与响应。

### 适用场景总结

| 应用场景 | 核心价值 |
| :--- | :--- |
| **企业RAG与知识库建设** | 为RAG系统提供高质量、语义完整的“数据燃料”，提升检索和问答精度 |
| **Agent应用开发** | 作为Agent的文档“眼睛”，让AI能准确理解并处理PDF合同、财报等复杂文件 |
| **金融、法律、科研等专业领域** | 零差错还原复杂表格、公式和排版，满足高精度要求，提升业务流程自动化效率 |

### 处理日常文档介绍

关于 TextIn xParse 对特定格式的处理，情况是这样的：**对于你列出的大部分格式，xParse 都支持作为输入进行解析，但也有一些是作为输出格式，或者暂未明确支持。**

### 📄 格式支持情况与示例

| 格式 | 是否支持输入 | 处理方式与备注 |
| :--- | :--- | :--- |
| **PDF** | ✅ 是 | **原生核心支持**。可处理扫描件、加密PDF等，并可导出为Markdown/JSON。 |
| **Docx** | ✅ 是 | **原生支持**。属于`Word`格式，可直接解析并结构化输出。 |
| **HTML** | ✅ 是 | **原生支持**。可直接作为输入文件进行解析。 |
| **TXT** | ✅ 是 | **原生支持**。作为纯文本格式，可直接输入解析。 |
| **CSV** | ✅ 是 | **原生支持**。属于`Excel`格式之一，xParse明确支持`xlsx`和`csv`文件的解析。 |
| **Markdown** | ✅ 是 | 支持作为**输入**，但更是其**核心输出格式**之一，用于将复杂文档内容标准化。 |
| **JSON** | ❌ 主要作为**输出** | xParse的核心功能是**将文档解析并输出为结构化JSON**，同时，也支持通过API以JSON格式传入参数。 |
| **XML** | ❌ 未明确提及 | 在现有资料中，**未找到xParse支持XML作为输入的直接证据**。 |

### 📝 简单的使用示例

TextIn xParse 提供了多种调用方式，这里展示最核心的 API 调用逻辑。

*   **上传本地文件并保存结果（Python 逻辑示意）**：这是最基本的使用方式，将本地文件发送给 API，并将返回的 Markdown 和 JSON 结果保存下来。

```python
import requests
import json

# 1. 配置您的认证信息
app_id = "您的AppId"
secret_code = "您的SecretCode"
api_url = "https://api.textin.com/ai/service/v1" # 请以官方实际 endpoint 为准

headers = {
    "x-ti-app-id": app_id,
    "x-ti-secret-code": secret_code,
    "Content-Type": "application/octet-stream",  # 上传文件用此格式
}

# 2. 指定要解析的本地文件
file_path = "您的文档.pdf"

# 3. 发起请求
with open(file_path, "rb") as f:
    response = requests.post(api_url, headers=headers, data=f)

if response.status_code == 200:
    json_response = response.json()
    # 保存完整的 JSON 结果
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(json_response, f, ensure_ascii=False, indent=2)

    # 如果返回结果中包含 Markdown，单独保存
    if "result" in json_response and "markdown" in json_response["result"]:
        markdown_content = json_response["result"]["markdown"]
        with open("result.md", "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print("解析完成！Markdown 和 JSON 结果已保存。")
else:
    print(f"解析失败，状态码：{response.status_code}")
```

*   **解析位于网络 URL 的文件**：xParse 也支持直接传入一个文件的 URL 地址进行解析，这在处理网络资源时非常方便。

```python
import requests
import json

# 认证信息同上...
headers = {
    "x-ti-app-id": app_id,
    "x-ti-secret-code": secret_code,
    "Content-Type": "text/plain",  # 发送 URL 使用此格式
}

# 要解析的文件的网络地址
file_url = "https://example.com/您的文档.pdf"

# 直接发送 URL 字符串
response = requests.post(api_url, headers=headers, data=file_url)

# 后续处理方式与上例相同...
```

### 💎 关键要点

1.  **格式支持全面**：TextIn xParse 几乎覆盖了你列举的所有主流文档格式，特别适合处理**PDF、Word、Excel、PPT、图片和HTML**等复杂文档。
2.  **输出标准化**：它的核心价值在于将各类非结构化文档**统一转化为 Markdown 或 JSON 格式**，这是为了给大语言模型（LLM）和 RAG 应用提供可直接使用的“燃料”。
3.  **关于 XML**：当前信息无法确认对 XML 格式的支持，如果需要处理 XML 文件，建议查阅官方最新文档或先将其转换为支持的格式。
4.  **集成方式多样**：除了直接 API 调用，它还提供了 **MCP Server、Coze、Dify 插件**以及**命令行工具 (CLI)** 等多种集成方式，方便在不同开发环境中使用。

# 更多文档解析前沿扩展
在成熟的商业工具之外，文档解析领域正经历一场由深度学习和大模型驱动的技术变革。当前的研究和实践，主要沿着**模块化流水线**和**端到端大模型**这两条技术路径演进，同时也有一些旨在提升交互体验的前沿探索。

### 两大技术流派演进

你可以把这两条路径看作两种不同的解题思路：

| 技术路径 | 核心思想 | 优势 | 挑战 |
| :--- | :--- | :--- | :--- |
| **模块化流水线 (Pipeline-based)** | 将复杂的解析任务拆解为**布局分析、内容识别、关系整合**等一系列独立的子任务，再像“流水线”一样串联起来。每个环节由最擅长的专家模型（如YOLO做布局检测，TrOCR做文字识别）负责。代表工具有**Docling、MinerU、RagFlow**等。 | 1. 流程清晰，每个环节可独立优化和替换。<br>2. 对于特定类型（如发票、表格）的精度可以做到非常高。 | 1. 存在**误差累积**问题，一个环节的错误会传递到下游。<br>2. 流程复杂，开发和维护成本高，像一条“精密但脆弱的流水线”。 |
| **端到端大模型 (End-to-End VLM)** | 使用一个强大的**多模态大语言模型（MLLM）**，直接将输入的文档图像“一步到位”地转化为结构化的Markdown或JSON格式。代表模型如**OvisOCR**、**百度Qianfan-OCR**。 | 1. 避免了流水线的误差累积问题，信息损失更少。<br>2. 能更好地理解文档的全局结构和视觉上下文，尤其擅长处理复杂表格、图表。 | 1. 对模型的推理能力要求极高，需要大量高质量的训练数据。<br>2. 模型尺寸和计算成本通常较高，对部署环境有要求。 |

### 🚀 处于探索阶段的前沿方向

#### 视觉提示理解 (Visual Prompting)
这种方案让AI模型能够“理解”用户在文档上做的标记，比如高亮、圈画等，并根据这些视觉提示来完成精准的问答或摘要。例如，你可以用鼠标高亮一段文字，然后问模型“总结一下这部分内容”。这比单纯的文字指令更加直观和高效。

#### 基于图的文档表示学习
通过将文档中的标题、段落、表格等元素视为节点，将元素之间的层级和引用关系视为边，构建一个“文档关系图”，从而更好地理解文档的逻辑结构。这种方法试图超越传统的“文字+坐标”表示，捕捉更丰富的语义关系。

### 💎 总结与选择建议

文档解析技术正从“专家分工的流水线”走向“模型统一的智能体”，并且向着更自然的人机交互方向发展。

在做技术选型时，你可以参考这个思路：

1.  **若你追求稳定可控、对特定类型文档精度要求极高，且团队有维护复杂系统的能力**：选择**模块化流水线**（如Docling、MinerU），它们更成熟。
2.  **若你追求极致效果，希望模型能更好地理解文档全局结构，且对计算成本有一定容忍度**：可以重点关注**端到端大模型**（如OvisOCR、Qianfan-OCR），这代表着未来的趋势。

