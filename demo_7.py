import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
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
    model="z-ai/glm4.7",
    temperature=0.3,
)

parser = StrOutputParser()

chain = llm | parser
result = chain.invoke("请简要介绍 LangChain 输出解析层的作用")

print("StrOutputParser 解析后的字符串：")
print(result)
print("\n解析结果类型：", type(result))
