"""정비이력 관리 에이전트 그래프: SQLite 기반 차량·정비이력 CRUD 단일 에이전트."""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from tools import (
    add_maintenance_record,
    delete_maintenance_record,
    get_maintenance_history,
    list_vehicles,
    register_vehicle,
    update_maintenance_record,
)

load_dotenv()

llm = ChatBedrockConverse(
    model=os.environ.get("AGENT_MODEL") or "us.anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
    temperature=0,
)

maintenance_agent = create_agent(
    llm,
    [
        register_vehicle,
        list_vehicles,
        get_maintenance_history,
        add_maintenance_record,
        update_maintenance_record,
        delete_maintenance_record,
    ],
    system_prompt=(
        "당신은 차량 정비이력 관리 도우미입니다. "
        "차량번호 외의 소유자 개인정보(이름, 전화번호 등)는 절대 답변에 담지 마세요. "
        "정비이력 수정·삭제처럼 되돌리기 어려운 작업은 먼저 사용자에게 말로 재확인을 받은 뒤에만 "
        "confirm=True로 도구를 호출하세요. 조회 결과가 없거나 등록되지 않은 차량이면 지어내지 말고 "
        "그대로 안내하세요. 경고등 의미, 점검 주기, 자가정비, 고장 증상, 안전수칙 같은 차량 매뉴얼 지식은 "
        "당신의 담당이 아니니 스스로 설명을 만들어 답하지 마세요. 그런 질문이 섞여 있으면 정비이력 관련 "
        "부분만 답하고, 매뉴얼 관련 내용은 다른 담당자가 안내한다고만 언급하세요."
    ),
    name="maintenance_agent",
)


def get_text(message):
    """ChatBedrockConverse 응답 메시지에서 텍스트만 추출해 반환한다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


if __name__ == "__main__":
    question = "12가3456 차량 마지막 타이어 교체한지 얼마나 됬지?"
    result = maintenance_agent.invoke({"messages": [HumanMessage(question)]})
    print(get_text(result["messages"][-1]))
