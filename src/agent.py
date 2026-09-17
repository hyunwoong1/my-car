"""Supervisor 에이전트 그래프: 매뉴얼 검색·정비이력 관리·정비소 조회 3개 에이전트를 통합한다."""
import os
from contextlib import ExitStack
from typing import Optional

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph_supervisor import create_supervisor

from location_agent import location_agent
from maintenance_agent import maintenance_agent
from manual_search_agent import manual_search_agent
from tools import get_registered_vehicle_types
from tracer import FileTracer, get_text

load_dotenv()

supervisor_llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
    temperature=0,
)

# 차종 코드가 보통 "몸체타입_모델번호"(예: sedan_1600) 형식이라, 앞부분(몸체타입)에 대한
# 한국어 표현만 별도로 매핑한다. 모델 번호까지는 하드코딩하지 않으므로 새 차종이 등록돼도
# (예: sedan_2200) "세단"이라는 표현으로 자동 인식된다.
_BODY_TYPE_ALIASES = {"sedan": "세단", "suv": "SUV"}


def _describe_vehicle_type(vehicle_type: str) -> str:
    """프롬프트에 보여줄 차종 표기. 별칭이 있으면 "sedan_1600(세단)"처럼 한글 표현을 덧붙인다."""
    prefix = vehicle_type.split("_")[0].lower()
    alias = _BODY_TYPE_ALIASES.get(prefix)
    return f"{vehicle_type}({alias})" if alias else vehicle_type


def _build_supervisor_prompt_text() -> str:
    """차량 마스터에 실제 등록된 차종 목록을 조회해 프롬프트에 주입한다. 특정 차종을 프롬프트에
    직접 못박아두지 않으므로, 새 차종의 차량이 등록되면 다음 호출부터 바로 반영된다."""
    types = get_registered_vehicle_types()
    # "세단"/"SUV"처럼 사용자가 차종 코드 대신 흔히 쓰는 한글 표현을 프롬프트에 같이 보여줘야,
    # 질문에 "세단 타이어..."처럼 코드 없이 차종이 이미 언급된 경우에도 Supervisor가 이를
    # sedan_1600 같은 실제 코드와 같은 것으로 인식하고 다시 묻지 않는다.
    types_desc = ", ".join(_describe_vehicle_type(t) for t in types) if types else "(현재 등록된 차량 없음)"
    example_type = types[0] if types else "그 차종"

    return (
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
        f"반드시 명시해서 전달하라(예: \"{example_type} 차종의 타이어 점검 주기 알려줘\").\n"
        "- 한 질문에 여러 요구가 섞여 있으면 필요한 Agent를 순서대로 호출해 모두 처리한 뒤 하나의 답변으로 종합하라.\n"
        "- 정비·차량과 무관한 질문(잡담 등)은 어떤 Agent에게도 넘기지 말고, "
        "\"차량 매뉴얼, 정비이력, 근처 정비소 관련 질문을 도와드릴 수 있습니다\"처럼 직접 안내하라.\n"
        "- Agent에게 위임할 때는 \"~하겠습니다\" 같은 안내 문구 없이 도구만 곧바로 호출하라. "
        "사용자에게 보여줄 답변 텍스트는 필요한 Agent 호출이 모두 끝난 뒤 최종 답변에서만 작성하라.\n"
        "- 더 이상 호출할 Agent가 없으면(모든 위임이 끝났으면) 절대 빈 답변으로 끝내지 마라. "
        "Agent들이 알아낸 내용을 반드시 너 자신의 말로 요약·정리해 사용자에게 보여줄 최종 답변 텍스트를 "
        "작성하라. Agent의 답변이 이미 충분해 보여도 그 내용을 반드시 최종 답변에 다시 담아라.\n"
        "- 점검 주기, 교체 주기, km 수치처럼 차종마다 값이 다를 수 있는 매뉴얼 질문인데 차량번호도 차종도 "
        f"언급되지 않았다면, manual_search_agent에게 위임하기 전에 먼저 사용자에게 현재 등록된 차종({types_desc}) "
        "중 어느 것인지 되물어라. 증상 설명, 안전수칙, 자가정비 방법처럼 차종에 관계없이 답이 같거나 "
        "매뉴얼에 없는 차종(전기차 등)에 대한 질문은 되묻지 말고 바로 위임하라.\n"
        "- 질문에 \"(참고: ... 차종은 ...)\" 같은 문구가 있으면 그건 사용자에 대해 이미 확인된 정보이니 "
        "다시 묻지 말고 그 차종으로 간주해 바로 위임하라.\n"
        "- 이미 다른 Agent의 응답(예: maintenance_agent가 알려준 차종)에 필요한 정보가 나와 있다면, 그걸 "
        "사용자에게 다시 확인하거나 다른 Agent가 그 정보를 모른다고 해서 포기하지 마라. 다음 Agent에게 "
        f"위임할 때 반드시 그 정보를 구체적인 요청 문장에 직접 포함시켜라(예: \"{example_type} 차종의 엔진오일 "
        "교체 주기 알려줘\"). 위임 메시지가 막연하면 해당 Agent가 이해하지 못할 수 있다.\n"
        "- 어떤 Agent가 정보 부족을 이유로 되묻는 답을 돌려주더라도, 그 되묻는 문장을 그대로 최종 답변에 "
        "옮겨 사용자에게 넘기지 마라. 되묻는 이유가 된 정보(차종, 항목명 등)를 네가 이미 알고 있다면 "
        "(다른 Agent의 이전 응답에 나와 있다면) 사용자에게 묻지 말고 그 정보를 채워 즉시 같은 Agent에게 "
        "다시 위임하라. 정말로 너도 그 정보를 모를 때만 사용자에게 되물어라."
    )


def _supervisor_prompt(state):
    """create_supervisor의 prompt로 넘기는 콜러블. 호출될 때마다 차량 마스터를 다시 조회하므로,
    대화 중간에 새 차량이 등록돼도 다음 턴부터 바로 최신 차종 목록이 반영된다."""
    return [SystemMessage(content=_build_supervisor_prompt_text())] + state["messages"]


supervisor = create_supervisor(
    [manual_search_agent, maintenance_agent, location_agent],
    model=supervisor_llm,
    prompt=_supervisor_prompt,
)

# 단기/장기 기억을 SQLite 파일로 영속화한다(프로세스를 껐다 켜도 대화·기억이 유지됨).
# 경로는 환경변수로 오버라이드 가능 — evaluation/run_eval.py가 평가 전용 파일로 바꿔치기해서
# 반복 실행 때마다 운영 데이터(data/checkpoints.sqlite 등)를 건드리지 않고 깨끗하게 리셋한다.
CHECKPOINT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")
STORE_DB_PATH = os.environ.get("STORE_DB_PATH", "data/store.sqlite")

# SqliteSaver/SqliteStore는 컨텍스트 매니저(with 블록 전용)라서, 모듈 전체 수명 동안 열어두려면
# ExitStack으로 __enter__만 하고 명시적으로 닫지 않는다(프로세스 종료 시 정리됨).
_sqlite_stack = ExitStack()
checkpointer = _sqlite_stack.enter_context(SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH))  # 단기 기억
store = _sqlite_stack.enter_context(SqliteStore.from_conn_string(STORE_DB_PATH))  # 장기 기억
store.setup()  # SqliteSaver와 달리 SqliteStore는 최초 1회 setup()을 직접 호출해야 한다
app = supervisor.compile(checkpointer=checkpointer, store=store)

tracer = FileTracer("trace.jsonl")


def _detect_vehicle_type(text: str) -> Optional[str]:
    """사용자 발화에서 차종을 감지한다(되물음에 대한 답, "세단이야" 같은 응답에서 쓴다).
    차량 마스터에 실제 등록된 차종만 대상으로 하므로, 새 차종이 등록되면 자동으로 인식 대상이 된다."""
    lowered = text.lower()
    for vehicle_type in get_registered_vehicle_types():
        prefix = vehicle_type.split("_")[0].lower()
        alias = _BODY_TYPE_ALIASES.get(prefix, prefix).lower()
        if vehicle_type.lower() in lowered or prefix in lowered or alias in lowered:
            return vehicle_type
    return None


def stream_answer_tokens(content: str, thread_id: str, user_id: str, callbacks: Optional[list] = None):
    """Supervisor 그래프를 실행하며 최종 답변 텍스트 조각을 순서대로 yield한다.
    콘솔 데모(run())와 API의 스트리밍 엔드포인트가 이 제너레이터를 함께 쓴다."""
    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 25,
    }
    if callbacks:
        config["callbacks"] = callbacks

    for namespace, (chunk, metadata) in app.stream(
        {"messages": [HumanMessage(content=content)]},
        config,
        stream_mode="messages",
        subgraphs=True,
    ):
        # supervisor 자신도 create_react_agent로 만들어진 서브그래프라서, subgraphs=True로 그
        # 내부(node="agent")까지 열어야 실제 토큰 단위 청크가 올라온다(안 그러면 서브그래프가
        # 끝난 뒤 완성된 메시지 하나가 통째로 오는 것만 보여서 스트리밍처럼 보이지 않는다).
        # 다른 서브 에이전트(maintenance_agent, manual_search_agent) 네임스페이스와 도구 호출
        # 결과는 콘솔/응답 스트림에 안 보여주고 callbacks(트레이서 등)에만 남긴다.
        root = namespace[0].split(":")[0] if namespace else None
        if root != "supervisor" or metadata.get("langgraph_node") != "agent":
            continue
        # 스트리밍 청크는 type이 "ai"가 아니라 "AIMessageChunk"로 온다("ai"는 완성된 AIMessage용).
        if getattr(chunk, "type", None) != "AIMessageChunk" or getattr(chunk, "tool_calls", None):
            continue
        text = get_text(chunk)
        if text:
            yield text


def run(question: str, thread_id: str = "default", user_id: str = "me") -> None:
    """질문 하나를 Supervisor 그래프에 흘려보내며 supervisor의 최종 답변만 스트리밍한다.
    같은 thread_id로 다시 부르면 checkpointer 덕분에 이전 대화(되물음 등)를 이어받고,
    user_id별로 store에 남은 "마지막으로 확인된 차종"이 있으면 질문에 참고 문구로 덧붙인다."""
    remembered = store.get(("users", user_id), "vehicle_type")
    content = question
    if remembered and not _detect_vehicle_type(question):
        content = f"{question}\n(참고: 이 사용자가 이전에 확인한 차종은 {remembered.value['vehicle_type']}입니다.)"

    print(f"질문: {question}")
    for text in stream_answer_tokens(content, thread_id, user_id, callbacks=[tracer]):
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
