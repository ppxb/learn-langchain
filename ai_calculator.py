import os
import re

from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("未读取到 API_KEY")

llm = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="deepseek-ai/deepseek-v4-flash",
    temperature=0.3,
)

calc_tool = PythonREPLTool()
WINDOW_SIZE = 2

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一名友好的个人助手，规则如下：
       1. 能记住最近{window_size}轮对话内容，用简单语言解答问题；
       2. 如果问题包含数学计算（如加减乘除、公式、数值运算），先调用计算工具得到结果，再用自然语言解释；
       3. 非计算问题直接回答，记得结合历史对话上下文。""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


def judge_and_calc(inputs):
    """
    核心逻辑：
    1. 检测用户问题是否包含数学计算需求
    2. 是：调用PythonREPLTool计算，再结合LLM生成回答
    3. 否：直接用LLM回答
    """

    user_input = inputs["input"]
    chat_history = inputs["chat_history"]

    calc_pattern = r"(\+|\-|\×|\*|÷|/|=|计算|求和|求差|平方|立方|多少|等于)"
    is_calc_needed = bool(re.search(calc_pattern, user_input))

    if is_calc_needed:
        try:
            calc_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", user_input)
            if not calc_expr:
                calc_result = "未识别到可计算的表达式"
            else:
                calc_result = calc_tool.run(calc_expr)
        except Exception as e:
            calc_result = f"计算出错：{str(e)}"

        # 步骤2：构造包含计算结果的提示，让LLM生成自然语言回答
        enhanced_input = f"""
                用户问题：{user_input}
                计算过程/结果：{calc_result}
                请结合计算结果，用简单易懂的语言回答用户问题，同时参考历史对话：{chat_history}
                """
        inputs["input"] = enhanced_input
    return inputs


# -------------------------- 4. 窗口记忆实现（仅保留最近N轮） --------------------------
# 会话存储：key=session_id，value=InMemoryChatMessageHistory
window_memory_store = {}


def get_window_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取带窗口限制的会话历史，自动截断超出长度的消息"""
    if session_id not in window_memory_store:
        window_memory_store[session_id] = InMemoryChatMessageHistory()

    history = window_memory_store[session_id]
    # 截断逻辑：保留最近WINDOW_SIZE轮（每轮2条消息：Human+AI）
    total_messages = len(history.messages)
    if total_messages > 2 * WINDOW_SIZE:
        history.messages = history.messages[-2 * WINDOW_SIZE :]
    return history


chain = RunnableLambda(judge_and_calc) | prompt | llm

chain_with_window_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_window_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output",
)
