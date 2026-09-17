"""Supervisor 그래프를 API로 노출한다. CLAUDE.md 제출 규약: POST /query로 question을 받고
answer, contexts, trace 세 키로 구조화된(Pydantic) 응답을 돌려준다."""
import uuid
from typing import Optional

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import app as graph_app
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


# 실행(mini-pjt 루트에서, src/ 모듈들이 flat import라 PYTHONPATH=src로 잡아줘야 한다):
#   PowerShell: $env:PYTHONPATH = "src"; uvicorn api_server:api --port 8000
# 테스트: Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType "application/json" -Body '{"question": "세단 타이어 공기압은 얼마나 자주 점검해야 해?"}'
