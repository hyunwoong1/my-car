"""Supervisor 에이전트 그래프: 매뉴얼 검색·정비이력 관리·정비소 조회 3개 에이전트를 통합한다."""
from typing import Optional

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor

from location_agent import location_agent
from maintenance_agent import maintenance_agent
from manual_search_agent import manual_search_agent
from tracer import FileTracer, get_text

load_dotenv()

supervisor_llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-6",
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
    "\"차량 매뉴얼, 정비이력, 근처 정비소 관련 질문을 도와드릴 수 있습니다\"처럼 직접 안내하라.\n"
    "- Agent에게 위임할 때는 \"~하겠습니다\" 같은 안내 문구 없이 도구만 곧바로 호출하라. "
    "사용자에게 보여줄 답변 텍스트는 필요한 Agent 호출이 모두 끝난 뒤 최종 답변에서만 작성하라.\n"
    "- 더 이상 호출할 Agent가 없으면(모든 위임이 끝났으면) 절대 빈 답변으로 끝내지 마라. "
    "Agent들이 알아낸 내용을 반드시 너 자신의 말로 요약·정리해 사용자에게 보여줄 최종 답변 텍스트를 "
    "작성하라. Agent의 답변이 이미 충분해 보여도 그 내용을 반드시 최종 답변에 다시 담아라.\n"
    "- 점검 주기, 교체 주기, km 수치처럼 차종마다 값이 다를 수 있는 매뉴얼 질문인데 차량번호도 차종도 "
    "언급되지 않았다면, manual_search_agent에게 위임하기 전에 먼저 사용자에게 세단(sedan_1600)인지 "
    "SUV(suv_2000d)인지 되물어라. 증상 설명, 안전수칙, 자가정비 방법처럼 차종에 관계없이 답이 같거나 "
    "매뉴얼에 없는 차종(전기차 등)에 대한 질문은 되묻지 말고 바로 위임하라.\n"
    "- 질문에 \"(참고: ... 차종은 ...)\" 같은 문구가 있으면 그건 사용자에 대해 이미 확인된 정보이니 "
    "다시 묻지 말고 그 차종으로 간주해 바로 위임하라."
)

supervisor = create_supervisor(
    [manual_search_agent, maintenance_agent, location_agent],
    model=supervisor_llm,
    prompt=SUPERVISOR_PROMPT,
)

checkpointer = InMemorySaver()  # 단기 기억: 같은 thread_id 안에서 대화 흐름(되물음 -> 답변)을 이어간다
store = InMemoryStore()  # 장기 기억: user_id별로 마지막에 확인된 차종을 기억해 다음에 재사용한다
app = supervisor.compile(checkpointer=checkpointer, store=store)

tracer = FileTracer("trace.jsonl")

VEHICLE_TYPES = ("sedan_1600", "suv_2000d")


def _detect_vehicle_type(text: str) -> Optional[str]:
    """사용자 발화에서 차종을 감지한다(되물음에 대한 답, "세단이야" 같은 응답에서 쓴다)."""
    if "sedan_1600" in text or "세단" in text:
        return "sedan_1600"
    if "suv_2000d" in text or "suv" in text.lower():
        return "suv_2000d"
    return None


def run(question: str, thread_id: str = "default", user_id: str = "me") -> None:
    """질문 하나를 Supervisor 그래프에 흘려보내며 supervisor의 최종 답변만 스트리밍한다.
    같은 thread_id로 다시 부르면 checkpointer 덕분에 이전 대화(되물음 등)를 이어받고,
    user_id별로 store에 남은 "마지막으로 확인된 차종"이 있으면 질문에 참고 문구로 덧붙인다."""
    remembered = store.get(("users", user_id), "vehicle_type")
    content = question
    if remembered and not _detect_vehicle_type(question):
        content = f"{question}\n(참고: 이 사용자가 이전에 확인한 차종은 {remembered.value['vehicle_type']}입니다.)"

    print(f"질문: {question}")
    for namespace, (chunk, metadata) in app.stream(
        {"messages": [HumanMessage(content=content)]},
        {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
            "callbacks": [tracer],
            "recursion_limit": 25,
        },
        stream_mode="messages",
        subgraphs=True,
    ):
        # supervisor 자신도 create_react_agent로 만들어진 서브그래프라서, subgraphs=True로 그
        # 내부(node="agent")까지 열어야 실제 토큰 단위 청크가 올라온다(안 그러면 서브그래프가
        # 끝난 뒤 완성된 메시지 하나가 통째로 오는 것만 보여서 스트리밍처럼 보이지 않는다).
        # 다른 서브 에이전트(maintenance_agent, manual_search_agent) 네임스페이스와 도구 호출
        # 결과는 trace.jsonl에만 남기고, 콘솔에는 supervisor가 생성하는 답변 텍스트만 스트리밍한다.
        root = namespace[0].split(":")[0] if namespace else None
        if root != "supervisor" or metadata.get("langgraph_node") != "agent":
            continue
        # 스트리밍 청크는 type이 "ai"가 아니라 "AIMessageChunk"로 온다("ai"는 완성된 AIMessage용).
        if getattr(chunk, "type", None) != "AIMessageChunk" or getattr(chunk, "tool_calls", None):
            continue
        text = get_text(chunk)
        if text:
            print(text, end="", flush=True)
    print()

    detected = _detect_vehicle_type(question)
    if detected:
        store.put(("users", user_id), "vehicle_type", {"vehicle_type": detected})


if __name__ == "__main__":
    # 1턴: 차종을 안 밝혀 supervisor가 되물어야 하는 질문
    run("타이어 공기압은 얼마나 자주 점검해야 해?", thread_id="demo")
    # 2턴: 같은 thread_id로 이어서 답하면(단기 기억) 되물음에 대한 답으로 이해하고,
    # 이 차종은 store에 남아(장기 기억) 이후 다른 대화에서도 참고된다.
    run("세단이야", thread_id="demo")
