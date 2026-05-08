import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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

prompt_template = PromptTemplate(
    input_variables=["user_role", "subject"],
    template="请给{user_role}写一段50字左右的{subject}学习建议，语言简洁实用，分2个小要点。",
)

formatted_prompt = prompt_template.format(user_role="高校学生", subject="LangChain")

print("格式化后的提示词：")
print(formatted_prompt)

result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])

print("\n生成的学习建议：")
print(result.content)
