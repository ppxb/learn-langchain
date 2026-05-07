import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("API_KEY not set")

chat_model = ChatOpenAI(
    api_key=SecretStr(API_KEY), base_url=BASE_URL, model="z-ai/glm4.7", temperature=0.3
)

messages = [
    {
        "role": "system",
        "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。",
    },
    {"role": "user", "content": "请用3句话解释什么是LangChain？"},
]

result = chat_model.invoke(messages)

print("ChatModel回复：")
print(result.content)
