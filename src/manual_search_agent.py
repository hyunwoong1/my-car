"""메인 에이전트 그래프: 차량 매뉴얼 검색(RAG) 단일 에이전트."""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from tools import search_vehicle_manual
from tracer import get_text

load_dotenv()

llm = ChatBedrockConverse(
    model=os.environ.get("AGENT_MODEL") or "us.anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
    temperature=0,
)

manual_search_agent = create_agent(
    llm,
    [search_vehicle_manual],
    system_prompt=(
        "당신은 차량 매뉴얼 안내 도우미입니다. "
        "반드시 도구 검색 결과에 근거해 답하고, 검색 결과 내에 질문과 관련이 없거나 근거가 없으면 모른다고 답하세요. "
        "질문이나 이전 대화에서 차종이 확인되면 search_vehicle_manual 호출 시 "
        "반드시 vehicle_type으로 지정해 그 차종의 매뉴얼만 검색하세요. 차종을 모르는 채로 검색해 다른 차종의 "
        "내용이 섞여 있다면, 그걸 마치 일반적인 내용인 것처럼 답하지 말고 어느 차종 내용인지 밝히거나 "
        "차종을 알려달라고 요청하세요.\n"
        "질문이나 위임 메시지에 이미 점검 항목명(예: 엔진오일 교체, 타이어 위치 교환)이나 정비이력 같은 "
        "배경 정보가 함께 주어졌더라도, 그 정보만으로 답을 짐작해서 만들지 마세요. 매뉴얼 내용에 대해 "
        "답하려면 반드시 그 항목으로 search_vehicle_manual을 최소 한 번 호출한 뒤에만 답하세요. 도구를 "
        "호출하지 않고 포기하거나 스스로 아는 내용으로 답하는 것은 금지입니다."
    ),
    name="manual_search_agent",
)


if __name__ == "__main__":
    question = "내 오토바이엔 어떤 연료를 넣어야해?"
    result = manual_search_agent.invoke({"messages": [HumanMessage(question)]})
    print(get_text(result["messages"][-1]))
