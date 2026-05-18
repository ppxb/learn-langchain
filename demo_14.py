import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
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


class TemperatureConvertInput(BaseModel):
    temperature: float = Field(description="需要转换的温度值，例如37.0")
    from_unit: str = Field(description="原始温度单位，只能是celsius或fahrenheit")


@tool(args_schema=TemperatureConvertInput)
def temperature_converter(temperature: float, from_unit: str) -> str:
    """温度单位转换工具"""

    if from_unit not in ["celsius", "fahrenheit"]:
        return f"错误：单位'{from_unit}'不合法"

    if from_unit == "celsius":
        fahrenheit = temperature * 9 / 5 + 32
        return f"{temperature}摄氏度 = {fahrenheit:.2f}华氏度"

    if from_unit == "fahrenheit":
        celsius = (temperature - 32) * 5 / 9
        return f"{temperature}华氏度 = {celsius:.2f}摄氏度"


tools = [temperature_converter]

system_prompt = """
你是一名专业温度转换助手，只能使用temperature_converter工具完成计算。
"""

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt, debug=True)

if __name__ == "__main__":
    query = "将37摄氏度转换为华氏度"
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    print("\n最终结果：")
    print(response["messages"][-1].content)
