import os

from dotenv import load_dotenv
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

chat_history = []


def chat_with_memory(user_input):
    prompt = "你是友好的助手，结合历史对话回答：\n"
    for msg in chat_history:
        prompt += f"{msg['role']}: {msg['content']}\n"

    prompt += f"用户: {user_input}\n"

    response = llm.invoke(prompt).content

    chat_history.append({"role": "用户", "content": user_input})
    chat_history.append({"role": "AI", "content": response})
    return response


print(chat_with_memory("我叫小明"))
print(chat_with_memory("我刚才叫什么名字？"))
