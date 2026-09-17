"""Supervisor 에이전트 그래프: 매뉴얼 검색·정비이력 관리·정비소 조회 3개 에이전트를 통합한다."""
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph_supervisor import create_supervisor

from location_agent import location_agent
from maintenance_agent import maintenance_agent
from manual_search_agent import manual_search_agent
from tracer import FileTracer, get_text

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
    "\"차량 매뉴얼, 정비이력, 근처 정비소 관련 질문을 도와드릴 수 있습니다\"처럼 직접 안내하라.\n"
    "- Agent에게 위임할 때는 \"~하겠습니다\" 같은 안내 문구 없이 도구만 곧바로 호출하라. "
    "사용자에게 보여줄 답변 텍스트는 필요한 Agent 호출이 모두 끝난 뒤 최종 답변에서만 작성하라.\n"
    "- 더 이상 호출할 Agent가 없으면(모든 위임이 끝났으면) 절대 빈 답변으로 끝내지 마라. "
    "Agent들이 알아낸 내용을 반드시 너 자신의 말로 요약·정리해 사용자에게 보여줄 최종 답변 텍스트를 "
    "작성하라. Agent의 답변이 이미 충분해 보여도 그 내용을 반드시 최종 답변에 다시 담아라."
)

supervisor = create_supervisor(
    [manual_search_agent, maintenance_agent, location_agent],
    model=supervisor_llm,
    prompt=SUPERVISOR_PROMPT,
)
app = supervisor.compile()

tracer = FileTracer("trace.jsonl")


if __name__ == "__main__":
    question = "12가3456 정비이력 보고 관련 매뉴얼도 같이 알려줘"
    for namespace, (chunk, metadata) in app.stream(
        {"messages": [HumanMessage(content=question)]},
        {"callbacks": [tracer], "recursion_limit": 25},
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
