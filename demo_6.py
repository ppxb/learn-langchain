import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_classic.document_loaders import srt
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

chat_model = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="z-ai/glm4.7",
    temperature=0.3,
)

with open("learning_method_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)


class DifficultyExampleSelector(BaseExampleSelector):
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def add_example(self, example: Dict[str, str]):
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[Dict]:
        target_difficulty = input_variables.get("difficulty", "easy")
        return [ex for ex in self.examples if ex.get("difficulty") == target_difficulty]


example_selector = DifficultyExampleSelector(examples=examples)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["subject", "difficulty", "method"],
        template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n",
    ),
    example_separator="\n",
    prefix="少样本提示：",
    suffix="参考以上示例，回答：\n学科：{new_subject}\n难度：{difficulty}\n学习方法：",
    input_variables=["new_subject", "difficulty"],
)

formatted_prompt_easy = few_shot_prompt.format(
    new_subject="LangChain", difficulty="easy"
)

print("入门级少样本提示词：")
print(formatted_prompt_easy)

result_easy = chat_model.invoke([{"role": "user", "content": formatted_prompt_easy}])
print("\n入门级学习方法：")
print(result_easy.content)

formatted_prompt_hard = few_shot_prompt.format(
    new_subject="LangChain", difficulty="hard"
)
print("\n进阶级少样本提示词：")
print(formatted_prompt_hard)
result_hard = chat_model.invoke([{"role": "user", "content": formatted_prompt_hard}])
print("\n进阶级学习方法：")
print(result_hard.content)
