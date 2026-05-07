import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, SecretStr

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("API_KEY not set")

llm = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="z-ai/glm4.7",
    temperature=0.3,
)


class WorkflowState(BaseModel):
    user_role: str | None = None
    original_advice: str | None = None
    simplified_advice: str | None = None
    english_advice: str | None = None


def generate_advice(state: WorkflowState):
    prompt = f"给{state.user_role}写一段50字左右的LangChain学习建议。"
    result = llm.invoke(prompt)
    return {"original_advice": result.content}


def simplify_advice(state: WorkflowState):
    prompt = f"把下面的学习建议精简到30字以内:{state.original_advice}"
    result = llm.invoke(prompt)
    return {"simplified_advice": result.content}


def translate_advice(state: WorkflowState):
    prompt = f"把下main的精简建议翻译成英文:{state.simplified_advice}"
    result = llm.invoke(prompt)
    return {"english_advice": result.content}


workflow = StateGraph(WorkflowState)

workflow.add_node("generate", generate_advice)
workflow.add_node("simplify", simplify_advice)
workflow.add_node("translate", translate_advice)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "simplify")
workflow.add_edge("simplify", "translate")
workflow.add_edge("translate", END)


app = workflow.compile()

result = app.invoke({"user_role": "高校学生"})

print("原始学习建议：")
print(result["original_advice"])
print("\n精简后学习建议：")
print(result["simplified_advice"])
print("\n翻译后学习建议：")
print(result["english_advice"])
