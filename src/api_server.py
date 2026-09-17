"""Supervisor 그래프를 API로 노출한다. CLAUDE.md 제출 규약: POST /query로 question을 받고
answer, contexts, trace 세 키로 구조화된(Pydantic) 응답을 돌려준다.
스트리밍이 필요한 클라이언트를 위해 POST /query/stream도 함께 제공한다(text/event-stream, SSE)."""
import json
import uuid
from typing import Iterator, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import app as graph_app, stream_answer_tokens
from tracer import RequestRecorder, get_text

api = FastAPI()


class QueryRequest(BaseModel):
    question: str


class TraceStep(BaseModel):
    type: str  # "agent"(위임) 또는 "tool"(도구 호출)
    name: str
    args: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    trace: list[TraceStep]


@api.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    rec = RequestRecorder()
    result = graph_app.invoke(
        {"messages": [HumanMessage(content=req.question)]},
        {
            # 요청마다 새 대화(단기 기억은 안 이어받음)로 처리하되, user_id는 고정해
            # 장기 기억(마지막으로 확인된 차종 등)은 API 호출 사이에도 유지되게 한다.
            "configurable": {"thread_id": str(uuid.uuid4()), "user_id": "default"},
            "callbacks": [rec],
            "recursion_limit": 25,
        },
    )
    answer = get_text(result["messages"][-1])
    return QueryResponse(
        answer=answer,
        contexts=rec.contexts,
        trace=[TraceStep(**step) for step in rec.trace],
    )


def _sse_event(event: str, data: dict) -> str:
    """SSE(text/event-stream) 한 이벤트를 만든다. data는 항상 JSON으로 인코딩해, 답변 조각에
    줄바꿈이 섞여 있어도 "data: ...\\n\\n" 한 덩어리로 안전하게 보낸다."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@api.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """POST /query와 같은 입력을 받아 최종 답변을 토큰 단위 SSE로 스트리밍한다.
    답변 조각마다 event: token으로 {"text": "..."}을 보내고, 다 끝나면 event: done으로
    contexts/trace를 한 번에 보낸다(/query와 같은 세 가지 정보를 스트리밍용으로 나눠 보내는 것)."""
    def event_source() -> Iterator[str]:
        rec = RequestRecorder()
        for text in stream_answer_tokens(
            req.question,
            thread_id=str(uuid.uuid4()),
            user_id="default",
            callbacks=[rec],
        ):
            yield _sse_event("token", {"text": text})
        yield _sse_event("done", {
            "contexts": rec.contexts,
            "trace": rec.trace,
        })

    return StreamingResponse(event_source(), media_type="text/event-stream")


# 실행(mini-pjt 루트에서, src/ 모듈들이 flat import라 PYTHONPATH=src로 잡아줘야 한다):
#   PowerShell: $env:PYTHONPATH = "src"; uvicorn api_server:api --port 8000
# 테스트(일반): Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType "application/json" -Body '{"question": "세단 타이어 공기압은 얼마나 자주 점검해야 해?"}'
# 테스트(스트리밍): curl -N -X POST http://localhost:8000/query/stream -H "Content-Type: application/json" -d '{"question": "세단 타이어 공기압은 얼마나 자주 점검해야 해?"}'
