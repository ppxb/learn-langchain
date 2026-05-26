import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
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

order_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是智能客服，负责解答用户的订单查询问题。"),
        (
            "human",
            "用户问题：{query}\n请引导用户提供订单号，并告知查询流程：1. 提供订单号；2. 系统验证；3. 反馈订单状态。",
        ),
    ]
)

order_chain = order_prompt | llm | StrOutputParser()

refund_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是智能客服，负责解答用户的退货款问题。"),
        (
            "human",
            "用户问题：{query}\n请说明退款流程：1. 申请退款（订单页面点击退款）；2. 等待审核（1-3个工作日）；3. 退款到账（原路返回，3-5个工作日）。如果用户问退款进度，引导提供退款申请单号。",
        ),
    ]
)

refund_chain = refund_prompt | llm | StrOutputParser()

warranty_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是智能客服，负责解答产品保修政策问题。"),
        (
            "human",
            "用户问题：{query}\n请说明保修政策：本产品保修期限为1年，保修范围包括质量问题（非人为损坏），保修流程：1. 联系客服；2. 提供购买凭证；3. 寄回检测维修。",
        ),
    ]
)

warranty_chain = warranty_prompt | llm | StrOutputParser()

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是路由选择器，需根据用户问题判断所属场景，仅输出以下标准化标识之一：
- order：订单查询相关（含订单状态、订单号）
- refund：退货款相关（含退款进度、退款申请）
- warranty：保修相关（含维修、售后保障）
- default：以上均不匹配
无需输出任何其他内容，仅返回标识字符串。
""",
        ),
        ("human", "用户问题：{query}"),
    ]
)

router_chain = router_prompt | llm | StrOutputParser()

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是智能客服。当遇到无法解答的问题时，请礼貌地告知用户你暂时无法处理该问题，并引导用户重新描述具体问题，或提供联系人工客服的方式（工作时间：9:00-18:00）。语气要友善、专业。",
        ),
        ("human", "用户问题：{query}\n请生成合适的回复。"),
    ]
)

default_chain = default_prompt | llm | StrOutputParser()

full_router_chain = RunnableLambda(lambda x: x) | (
    RunnableBranch(
        (lambda x: x["scene"] == "order", order_chain),
        (lambda x: x["scene"] == "refund", refund_chain),
        (lambda x: x["scene"] == "warranty", warranty_chain),
        default_chain,
    )
).with_config(run_name="full_router_chain")


def process_query(query: str):
    scene = router_chain.invoke({"query": query})
    return full_router_chain.invoke({"query": query, "scene": scene})


test_queries = [
    "我的订单什么时候发货？",
    "怎么申请退款呀？",
    "这个产品保修多久？",
    "你们家有什么新品？",  # 无法匹配，触发默认链
]

for query in test_queries:
    print(f"\n用户问题：{query}")
    print("客服回复：", process_query(query))
