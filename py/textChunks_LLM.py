from langchain_openai import ChatOpenAI              # 保持原导入
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# 用 ChatOpenAI 指向 DeepSeek 的 API 地址
llm = ChatOpenAI(
    model="deepseek-chat",                            # DeepSeek 模型名
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),    # 读取 DeepSeek 的 Key
    openai_api_base="https://api.deepseek.com/v1",   # 关键：修改 base_url
    temperature=0
)

# 后续代码与方式一完全相同
template = """
你是一位文档分析专家。请将以下文本按语义主题切分为多个段落。
在每个语义边界处用 "---SPLIT---" 标记。
要求：每个段落围绕一个完整子主题；不切断完整句子；保持原文不变。

文本：
{text}

请输出切分后的文本：
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

text = """人工智能是计算机科学的重要分支。
机器学习是人工智能的核心方法。
深度学习使用多层神经网络处理复杂模式。
篮球是一项广受欢迎的运动。
NBA汇集了全世界顶尖运动员。
团队合作在篮球比赛中至关重要。"""

result = chain.invoke({"text": text})
chunks = [c.strip() for c in result.split("---SPLIT---") if c.strip()]

print(f"✅ 生成块数: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    print(f"--- 块 {i} ---\n{chunk}\n")