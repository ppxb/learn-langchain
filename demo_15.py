import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.agent_toolkits import FileManagementToolkit
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
    model="deepseek-ai/deepseek-v4-flash",
    temperature=0.3,
)


toolkit = FileManagementToolkit(root_dir=".")
tools = toolkit.get_tools()

agent = create_agent(model=llm, tools=tools, debug=True)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请创建一个名为 llm诗词.txt 的文件，并在文件中写入一首原创七言绝句，主题围绕科技与人文的融合。",
            }
        ]
    }
)

print("\n任务执行完成！文件已写入。")
print("Agent最终输出：\n", response["messages"][-1].content)
