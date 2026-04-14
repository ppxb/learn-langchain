import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0.7
)


def demo_invoke():
    print("=" * 50)
    print("INVOKE 调用")

    messages = [
        SystemMessage(content="你是一个简洁的 AI 助手，回答不超过 50 字。"),
        HumanMessage(content="什么是 LangChain？")
    ]

    response: AIMessage = llm.invoke(messages)

    print(f"回答：{response.content}")
    print(f"Token 用量：{response.response_metadata.get('token_usage', {})}")


def demo_stream():
    print("\n" + "=" * 50)
    print("流式输出")

    messages = [
        HumanMessage(content="用三句话介绍一下 Python 语言的特点。")
    ]

    print("回答：", end="", flush=True)
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    # demo_invoke()
    demo_stream()

    print("\n✅ 运行完毕！请前往 https://smith.langchain.com 查看 'study' 项目的 Trace 记录。")
