"""메인 에이전트 그래프: 차량 매뉴얼 검색(RAG) 단일 에이전트."""
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from tools import search_vehicle_manual

load_dotenv()

llm = ChatBedrockConverse(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
)

manual_search_agent = create_agent(
    llm,
    [search_vehicle_manual],
    system_prompt=(
        "당신은 차량 매뉴얼 안내 도우미입니다. "
        "반드시 도구 검색 결과에 근거해 답하고, 검색 결과 내에 질문과 관련이 없거나 근거가 없으면 모른다고 답하세요. "
        "질문이나 이전 대화에서 차종(sedan_1600 또는 suv_2000d)이 확인되면 search_vehicle_manual 호출 시 "
        "반드시 vehicle_type으로 지정해 그 차종의 매뉴얼만 검색하세요. 차종을 모르는 채로 검색해 다른 차종의 "
        "내용이 섞여 있다면, 그걸 마치 일반적인 내용인 것처럼 답하지 말고 어느 차종 내용인지 밝히거나 "
        "차종을 알려달라고 요청하세요."
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
    question = "내 오토바이엔 어떤 연료를 넣어야해?"
    result = manual_search_agent.invoke({"messages": [HumanMessage(question)]})
    print(get_text(result["messages"][-1]))
