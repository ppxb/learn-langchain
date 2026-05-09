import os

from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("API_KEY not set")

chat_model = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="z-ai/glm4.7",
    temperature=0.3,
)

examples = [
    {
        "subject": "Python编程",
        "method": "核心目标：掌握基础语法和常用库；学习步骤：1. 学习变量、函数等基础语法 2. 实操小项目（如计算器） 3. 学习Pandas、Matplotlib库；注意事项：多动手实操，遇到错误及时调试。",
    },
    {
        "subject": "机器学习",
        "method": "核心目标：理解基础算法原理和应用场景；学习步骤：1. 复习数学基础（线性代数、概率） 2. 学习经典算法（线性回归、决策树） 3. 用Scikit-learn实操；注意事项：先理解原理，再动手实现，避免死记硬背。",
    },
]

example_template = """
学科：{subject}
学习方法：{method}
"""

# set example prompt
example_prompt = PromptTemplate(
    input_variables=["subject", "method"], template=example_template
)

# build few shots prompt
few_shots_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="学科：{new_subject}\n学习方法：",
    input_variables=["new_subject"],
)

# format prompt
formatted_prompt = few_shots_prompt.format(new_subject="LangChain")

print("少样本提示词：")
print(formatted_prompt)

result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])

print("\n生成的LangChain学习方法：")
print(result.content)
