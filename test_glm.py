import os

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
API_KEY = os.getenv("API_KEY", "sk-no-key-needed").strip()
BASE_URL = os.getenv("BASE_URL", "").strip()

llm = ChatOpenAI(
    model="glm5",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.3,
    default_headers={"User-Agent": "curl/8.0"},
)

print(llm.invoke("hi").content)
