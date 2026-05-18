import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("未读取到 API_KEY")

llm = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="deepseek-ai/deepseek-v4-flash",
    temperature=0.3,
)

calc_tool = PythonREPLTool()
WINDOW_SIZE = 2

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一名友好的个人助手，规则如下：
       1. 能记住最近{window_size}轮对话内容，用简单语言解答问题；
       2. 如果问题包含数学计算（如加减乘除、公式、数值运算），先调用计算工具得到结果，再用自然语言解释；
       3. 非计算问题直接回答，记得结合历史对话上下文。""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
