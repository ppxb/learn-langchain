import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

llm = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="deepseek-ai/deepseek-v4-flash",
    temperature=0.3,
)

sell_point_prompt = PromptTemplate.from_template(
    "从以下产品介绍中提取3个核心卖点，简洁列出：{product_intro}"
)

marketing_prompt = PromptTemplate.from_template(
    "针对{target_audience}，结合以下核心卖点，写一段朋友圈营销话术：{sell_points}"
)

# 现代化的 LCEL 链条
overall_chain = (
    {
        # 使用 StrOutputParser() 替代 lambda x: x.content
        "sell_points": sell_point_prompt | llm | StrOutputParser(),
        "target_audience": RunnablePassthrough(),
    }
    | marketing_prompt
    | llm
    | StrOutputParser()  # 让最终结果也是纯文本
)

input_data = {
    "product_intro": "这款无线耳机采用蓝牙5.3芯片...",
    "target_audience": "大学生群体...",
}

# 此时 result 直接就是字符串，不需要再 .content 了
result = overall_chain.invoke(input_data)
print("\n最终营销话术：")
print(result)
