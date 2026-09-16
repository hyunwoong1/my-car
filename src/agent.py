"""Supervisor 에이전트 그래프: 매뉴얼 검색·정비이력 관리·정비소 조회 3개 에이전트를 통합한다."""
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph_supervisor import create_supervisor

from agent_car_mgmt import maintenance_agent
from agent_location import location_agent
from agent_manual import manual_search_agent

load_dotenv()

supervisor_llm = ChatBedrockConverse(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
)

SUPERVISOR_PROMPT = (
    "너는 개인 차량 관리 서비스의 작업 분배자(Supervisor)다.\n"
    "\n"
    "[배분 기준]\n"
    "- 경고등 의미, 점검 주기, 자가정비 가능 항목, 고장 증상, 안전수칙 등 차량 매뉴얼 지식은 manual_search_agent\n"
    "- 차량 등록·목록 조회, 정비이력 등록·조회·수정·삭제는 maintenance_agent\n"
    "- 근처 정비소를 찾는 질문은 location_agent\n"
    "\n"
    "[규칙]\n"
    "- 직접 답을 지어내지 말고 반드시 담당 Agent를 통해 확인하라.\n"
    "- 차량번호만 언급되고 차종을 모르는 상태에서 매뉴얼 검색이 함께 필요하면, 먼저 maintenance_agent에게 "
    "그 차량번호의 차종(vehicle_type)을 확인한 뒤, manual_search_agent에게 위임할 때 확인된 차종을 "
    "반드시 명시해서 전달하라(예: \"sedan_1600 차종의 타이어 점검 주기 알려줘\").\n"
    "- 한 질문에 여러 요구가 섞여 있으면 필요한 Agent를 순서대로 호출해 모두 처리한 뒤 하나의 답변으로 종합하라.\n"
    "- 정비·차량과 무관한 질문(잡담 등)은 어떤 Agent에게도 넘기지 말고, "
    "\"차량 매뉴얼, 정비이력, 근처 정비소 관련 질문을 도와드릴 수 있습니다\"처럼 직접 안내하라."
)

supervisor = create_supervisor(
    [manual_search_agent, maintenance_agent, location_agent],
    model=supervisor_llm,
    prompt=SUPERVISOR_PROMPT,
)
app = supervisor.compile()


def get_text(message):
    """ChatBedrockConverse 응답 메시지에서 텍스트만 추출해 반환한다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


def run(question: str) -> None:
    """질문 하나를 Supervisor 그래프에 흘려보내며 호출된 에이전트/도구와 답변을 출력한다.
    supervisor 노드는 결정을 내릴 때마다 그때까지의 전체 메시지 목록을 다시 돌려주므로,
    이미 출력한 메시지는 id로 걸러내 한 번씩만 보여준다."""
    print(f"질문: {question}")
    seen_ids: set[str] = set()
    for event in app.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates",
        config={"recursion_limit": 25},
    ):
        for node, update in event.items():
            for m in (update or {}).get("messages", []):
                mid = getattr(m, "id", None)
                if mid is not None:
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                if getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        print(f"  [{node}] 도구 호출: {tc['name']}({tc.get('args', {})})")
                elif getattr(m, "content", None):
                    label = getattr(m, "name", None) or node
                    print(f"  [{label}] {get_text(m)}")


if __name__ == "__main__":
    run("12가3456 정비이력 보고 관련 매뉴얼도 같이 알려줘")
