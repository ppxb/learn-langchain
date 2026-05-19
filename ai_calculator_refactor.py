import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
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

tools = [PythonREPLTool()]

MAX_MESSAGES = 10


@before_model
def trim_history(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if len(messages) <= MAX_MESSAGES:
        return None

    first_message = messages[0]
    recent = messages[-MAX_MESSAGES:]
    if recent and recent[0].type != "human":
        recent = messages[-(MAX_MESSAGES - 1) :]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first_message,
            *recent,
        ]
    }


checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一名贴心的个人助手。"
        "如果用户提出涉及数学计算的问题，务必生成对应的 Python 代码并调用计算工具获取准确结果，"
        "不要自行猜算。最终以自然、友好的语言回复用户。"
    ),
    middleware=[trim_history],
    checkpointer=checkpointer,
)


def chat(user_input: str, thread_id: str = "default") -> str:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke({"messages": user_input}, config)
    return response["messages"][-1].content


if __name__ == "__main__":
    thread_id = "user_chat_001"
    print("=" * 54)
    print("  DeepSeek V4 智能助手 · LangChain 1.x")
    print("=" * 54)
    print("输入 'q' 退出\n")

    while True:
        try:
            user_input = input("你：").strip()
        except EOFError, KeyboardInterrupt:
            print("\n助手：再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in {"q", "quit", "exit"}:
            print("助手：再见！")
            break

        try:
            reply = chat(user_input, thread_id)
            print(f"助手：{reply}\n")
        except Exception as e:
            print(f"助手：[遇到了一点小麻烦] {e}\n")
