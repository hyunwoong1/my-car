"""메인 에이전트 그래프: 차량 매뉴얼 검색(RAG) 단일 에이전트."""
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from tools import search_vehicle_manual

load_dotenv()

llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
)

manual_search_agent = create_agent(
    llm,
    [search_vehicle_manual],
    system_prompt=(
        "당신은 차량 매뉴얼 안내 도우미입니다. "
        "반드시 도구 검색 결과에 근거해 답하고, 근거가 없으면 모른다고 답하세요."
    ),
    name="manual_search_agent",
)


def get_text(message):
    """ChatBedrockConverse 응답 메시지에서 텍스트만 추출해 반환한다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


if __name__ == "__main__":
    question = "디젤 SUV에서 DPF 경고등이 자주 뜨는데 어떻게 해야 해?"
    result = manual_search_agent.invoke({"messages": [HumanMessage(question)]})
    print(get_text(result["messages"][-1]))
