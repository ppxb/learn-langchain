import os

from dotenv import load_dotenv
from langchain_core.output_parsers import (
    BaseOutputParser,
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


class CustomToolParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        """将模型输出按 '工具名@核心功能@学习难度' 解析为字典"""
        text = text.strip().replace("\n", "").replace(" ", "")
        parts = text.split("@")
        if len(parts) != 3:
            raise ValueError(
                f"输出格式错误！需满足「工具名@核心功能@学习难度」，当前输出：{text}"
            )

        return {
            "tool_name": parts[0].strip(),
            "function": parts[1].strip(),
            "difficulty": parts[2].strip(),
        }

    def get_format_instructions(self) -> str:
        """生成提示词，引导模型按自定义格式输出"""
        return "请严格按照「工具名@核心功能@学习难度」格式输出，不添加多余内容。示例：LangSmith@全链路调试监控@中等"


parser = CustomToolParser()

prompt = PromptTemplate(
    template="请介绍1个LangChain开发工具。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

result = chain.invoke({})

print("自定义解析器解析结果：")
print(result)
print("解析结果类型：", type(result))
