import os

from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
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


full_memory_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是友好的对话助手，需基于完整的历史对话回答用户问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ]
)

base_chain = full_memory_prompt | llm
full_memory_store = {}


def get_full_memory_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in full_memory_store:
        full_memory_store[session_id] = InMemoryChatMessageHistory()
    return full_memory_store[session_id]


full_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_full_memory_history,
    input_messages_key="user_input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "user_0001"}}

response1 = full_memory_chain.invoke(
    {"user_input": "我叫小明，喜欢编程"}, config=config
)
print("助手回复1：", response1.content)

response2 = full_memory_chain.invoke(
    {"user_input": "我刚才说我喜欢什么？"}, config=config
)
print("助手回复2：", response2.content)

print("\n全量记忆的对话历史：")
for msg in get_full_memory_history("user_0001").messages:
    print(f"{msg.type}: {msg.content}")
