import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
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
    model="deepseek-ai/deepseek-v4-flash",
    temperature=0.3,
)


@tool
def weather_query(city: str):
    """查询指定城市天气"""
    weather_data = {
        "北京": "北京今日天气：晴，-2~8℃",
        "上海": "上海今日天气：多云，5~12℃",
        "广州": "广州今日天气：小雨，18~25℃",
    }
    return weather_data.get(city, f"暂无 {city} 数据")


tools = [weather_query]

agent = create_agent(model=llm, tools=tools, debug=True)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "北京今天的天气怎么样？"}]}
)

print("\n最终回答：")
print(response["messages"][-1].content)
