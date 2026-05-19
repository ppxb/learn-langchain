import os

from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
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
    temperature=0,
)

calc_tool = PythonREPLTool()

# 1. 绑定工具
llm_with_tools = llm.bind_tools([calc_tool])

# 2. 滚动窗口记忆
msg_trimmer = trim_messages(
    strategy="last",
    max_tokens=10,
    token_counter=len,
    allow_partial=False,
    start_on="human",
)

# 3. 构造智能体 LCEL 链与 Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名贴心的个人小助手。如果用户提出的问题涉及数学计算，请务必生成对应的代码并调用计算工具来获得准确结果，不要自己盲目计算。",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


def tool_chain(model_output):
    """拦截LLM的输出，如果是tool_calls则执行工具，否则直接返回文本"""
    if model_output.tool_calls:
        tool_call = model_output.tool_calls[0]
        result = calc_tool.run(tool_call["args"]["__arg1"])
        return llm.invoke(
            f"用户问题：{tool_call['args']['__arg1']}\n计算结果为：{result}\n请帮我用自然语言友好地告知用户。"
        )
    return model_output


chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: msg_trimmer.invoke(x["chat_history"])
    )
    | prompt
    | llm_with_tools
    | tool_chain
)

session_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


configurable_chain = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    session_id = "user_chat_001"
    print("===== 基于 LangChain 最新特性构建的智能助手 =====")
    print("✨ 特性：LLM 原生 Tool Calling + trim_messages 动态窗口管理")
    print("输入 'q' 退出对话\n")

    while True:
        user_input = input("你：")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("助手：再见！")
            break
        if not user_input.strip():
            continue

        try:
            response = configurable_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"助手：{response.content}\n")
        except Exception as e:
            print(f"助手：[遇到了一点小麻烦] {e}\n")
