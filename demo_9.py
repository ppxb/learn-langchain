import os

from dotenv import load_dotenv
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

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


class ToolInfo(BaseModel):
    tool_name: str = Field(description="LangChain开发工具的名称，如 LangSmith")
    function: str = Field(description="工具的核心功能，30字以内")
    difficulty: str = Field(description="学习难度，仅可选：简单 / 中等 / 复杂")


parser = PydanticOutputParser(pydantic_object=ToolInfo)

prompt = PromptTemplate(
    template="{user_input}，严格按照要求输出。\n{format_instructions}",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


chain = prompt | llm | parser

result = chain.invoke({"user_input": "请介绍1个 Python 开发工具"})

print("解析后的结构化数据（Pydantic 模型对象）：")
print(result)

print("字段校验 difficulty：", result.difficulty)

print("转化为字典：", result.model_dump())
