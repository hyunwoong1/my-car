# 개인 차량 관리 Agent

여러 대의 차량(차량번호 + 차종)을 관리하는 개인용 서비스. 경고등·이상 증상이 생겼을 때 매뉴얼 지식으로
바로 답하고, 내 차량의 정비이력을 직접 CRUD로 기록·조회하며, 근처 정비소도 찾아준다.

자세한 기획 배경은 [SERVICE.md](SERVICE.md) 참고.

## 아키텍처

Supervisor(`langgraph_supervisor.create_supervisor`)가 질문을 분류해 아래 3개 서브 에이전트로
라우팅하고, 여러 에이전트가 필요한 질문(예: "정비이력 보고 관련 매뉴얼도 같이 알려줘")은 순서대로
호출해 하나의 답으로 종합한다.

```
                         ┌─────────────────────┐
                question │      Supervisor      │ answer
              ──────────▶│    (src/agent.py)     │──────────▶
                         └──────────┬───────────┘
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                   ▼
        manual_search_agent   maintenance_agent    location_agent
        (RAG, 매뉴얼 검색)    (SQLite CRUD)         (지역 장소 검색)
```

- **manual_search_agent** (`src/manual_search_agent.py` + `src/retriever.py`) — 차종별 차량
  매뉴얼(경고등 의미, 점검 주기, 자가정비 가능 항목, 고장 증상, 안전수칙)을 BM25 + 벡터 검색
  하이브리드(`EnsembleRetriever`)로 찾는다.
- **maintenance_agent** (`src/maintenance_agent.py` + `src/tools.py`) — 차량 등록·목록 조회,
  정비이력 등록·조회·수정·삭제(SQLAlchemy ORM, `data/maintenance.db`).
- **location_agent** (`src/location_agent.py` + `src/tools.py`) — 지역명 + 업종 키워드로 정비소를
  찾는 범용 지역 검색(현재는 `data/local_places.json` 더미, 네이버 지역검색 API로 교체 가능한
  형태로 함수 시그니처만 유지).

## 폴더 구조

```
src/
  agent.py               Supervisor 그래프 (진입점)
  manual_search_agent.py  매뉴얼 검색 에이전트
  maintenance_agent.py    정비이력 관리 에이전트
  location_agent.py       정비소 조회 에이전트
  tools.py                8개 도메인 도구 (RAG 검색, 차량/정비이력 CRUD, 장소 검색)
  retriever.py            RAG 파이프라인 (청킹 → Chroma → BM25+벡터 하이브리드)
  tracer.py               파일 기반 트레이스 로깅 + API용 요청 레코더
  api_server.py           FastAPI 서버 (POST /query)
data/
  manuals/                차종별 매뉴얼 md (더미)
  maintenance.db          차량 마스터 + 정비이력 SQLite (seed.json으로 최초 시딩)
  local_places.json       지역 장소 더미 데이터
  chroma_db/              매뉴얼 임베딩 벡터스토어 (최초 실행 시 자동 생성)
evaluation/
  test_queries.csv        인-아웃 평가셋 12건 (positive/negative/edge/guardrail)
  eval_lib.py             LLM-as-Judge 심사 + RAGAS 스타일 지표(Faithfulness/Answer Relevancy/Context Relevance)
  run_eval.py             평가 실행 스크립트 (round2_*를 계속 덮어씀)
  round1_report.md        최초 베이스라인 리포트
  round2_report.md        최신 평가 리포트
web/
  index.html/styles.css/tokens.css/app.js   API 연동 웹 채팅 UI (개인용 앱 셸, Hallmark로 제작)
```

## 실행 방법

```bash
# 콘솔 데모 (질의응답 스트리밍 확인)
cd mini-pjt
python src/agent.py

# 자체 평가 (evaluation/maintenance_test.db를 매번 초기화 후 실행, round2_* 덮어씀)
python evaluation/run_eval.py
```

`.env`(레포 루트)에 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` 필요
(Amazon Bedrock, `ChatBedrockConverse` 사용). 에이전트/심사 모델은 기본값이 `claude-sonnet-4-6`이며,
`.env`에 `AGENT_MODEL` / `JUDGE_MODEL`을 넣으면 코드 수정 없이 다른 모델로 바꿀 수 있다.

## 웹 채팅 UI

`web/index.html`을 브라우저로 열면(정적 파일이라 별도 빌드 불필요) API 서버와 통신하는 채팅
화면을 쓸 수 있다. API 서버가 먼저 떠 있어야 하며, 기본 주소(`http://localhost:8000`)가 아니면
화면의 "API 주소 설정"에서 바꿀 수 있다.

## API 서버

`src/` 모듈들이 flat import라 `PYTHONPATH`를 `src`로 잡아준 뒤 `mini-pjt` 루트에서 uvicorn으로
띄운다.

```bash
cd mini-pjt
$env:PYTHONPATH = "src"; uvicorn api_server:api --port 8000
```

### 요청 예시

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "12가3456 차량 마지막 정비가 언제였어?", "thread_id": "demo-1"}'
```

`thread_id`는 선택 필드다. 같은 값을 계속 보내야 그 대화(되물음 → 답변 등)가 이어지고, 생략하면
매 요청이 새 대화로 처리된다(자세한 내용은 [메모리](#메모리) 참고).

### 응답 예시

`answer`(최종 답변), `contexts`(도구 호출 결과 원문 목록), `trace`(위임·도구 호출 순서) 세 키로
구조화되어 돌아온다(`api_server.py`의 Pydantic `QueryResponse`). 아래는 실제 서버를 띄워 위 요청을
보내 받은 응답이다.

```json
{
  "answer": "**12가3456** 차량의 마지막 정비 내역을 확인했습니다.\n\n- **정비 날짜:** 2025년 7월 15일\n- **정비 항목:** 엔진오일 교체\n\n혹시 해당 정비 이력에 대해 수정하거나 추가로 확인하고 싶은 내용이 있으시면 말씀해 주세요!",
  "contexts": [
    "차종: sedan_1600\n[6] 2025-07-15 - 엔진오일 교체 (비용: None원, 다음 점검 권장일: None)\n..."
  ],
  "trace": [
    {"type": "agent", "name": "maintenance_agent", "args": null},
    {"type": "tool", "name": "get_maintenance_history", "args": "{'vehicle_no': '12가3456'}"}
  ]
}
```

### 스트리밍 (`POST /query/stream`)

같은 입력을 받아 답변을 토큰 단위로 실시간 스트리밍하고 싶은 클라이언트를 위한 SSE
(`text/event-stream`) 엔드포인트. 답변이 생성되는 대로 `event: token`을 여러 번 보내고, 끝나면
`event: done`으로 `/query`와 같은 `contexts`/`trace`를 한 번에 보낸다. 그래프 실행 중 예외(예:
Bedrock 한도 초과)가 나면 연결이 그냥 끊기는 대신 `event: error`로 알려준다.

```bash
# curl (-N: 버퍼링 없이 바로바로 출력)
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "12가3456 차량 마지막 정비가 언제였어?", "thread_id": "demo-1"}'
```

```
event: token
data: {"text": "**12가3456** 차"}

event: token
data: {"text": "량의 마지막 정비 내역은..."}

...

event: done
data: {"contexts": ["차종: sedan_1600\n[6] 2025-07-15 - 엔진오일 교체 ..."], "trace": [{"type": "agent", "name": "maintenance_agent"}, {"type": "tool", "name": "get_maintenance_history", "args": "{'vehicle_no': '12가3456'}"}]}
```

그래프 실행 중 예외가 나면 `event: done` 대신 아래처럼 온다. 실제 예외 내용(예: Bedrock
한도 초과 메시지)은 `trace.jsonl`에만 남기고, 클라이언트에는 일반적인 오류 메시지만 보낸다
(`/query`는 같은 경우 502로 응답):

```
event: error
data: {"message": "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}
```

## 가드레일

- 매뉴얼/정비이력/정비소 조회 결과가 없으면 지어내지 않고 "확인 안 되는 정보"로 안내
- 차량번호 외 소유자 개인정보(이름·전화번호 등) 노출 금지
- 정비이력 삭제·수정은 사용자 재확인(`confirm=True`) 전까지 실행하지 않음
- 안전 관련 증상은 반드시 정비소 방문 권고 문구 포함
- 차종마다 값이 다를 수 있는 질문(점검 주기 등)인데 차종이 불명확하면, 현재 차량 마스터에 등록된
  차종 중 어느 것인지 먼저 되묻는다 (하드코딩된 차종명이 아니라 DB에서 조회해 프롬프트에 주입하므로,
  새 차종이 등록되면 자동으로 선택지에 반영됨). 이전 대화나 다른 세션에서 확인된 차종이 있어도
  세션을 넘어 다른 질문에 그대로 넘겨짚지 않는다 — 사용자가 차종이 다른 차량을 여러 대 갖고 있을
  수 있어서, 지금 이 질문에서 확인되지 않았다면 다시 묻는다

## 메모리

`SqliteSaver`(checkpointer)로 SQLite 파일(`data/checkpoints.sqlite`, `CHECKPOINT_DB_PATH` 환경변수로
경로 변경 가능)에 영속화되어 있어, 프로세스를 껐다 켜도 같은 `thread_id` 안의 대화(되물음 → 답변 등)는
이어받는다. 차종처럼 차량마다 다를 수 있는 정보는 세션을 넘어서는 장기 기억으로 남기지 않는다 —
다차량 사용자에게 이전에 확인한 차종을 다른 차량 질문에 잘못 적용할 위험이 있어서, 매번 대화(같은
`thread_id`) 안에서만 문맥을 이어받고 새 대화에서는 다시 확인한다.

## 평가

`evaluation/test_queries.csv`의 12개 케이스(positive 5 · negative 2 · edge 3 · guardrail 2)를
LLM-as-Judge(Pydantic 구조화 출력)로 채점하고, Faithfulness / Answer Relevancy / Context
Relevance를 RAGAS와 동일한 개념으로 직접 산출한다. 결과는 `evaluation/round*_report.md` 참고.

## 기술 스택

LangChain `create_agent`(서브 에이전트) + `langgraph_supervisor.create_supervisor`(멀티에이전트
오케스트레이션), Amazon Bedrock(`ChatBedrockConverse`, Claude Sonnet), Chroma + BM25 하이브리드
RAG, SQLAlchemy ORM, Pydantic 구조화 출력(API 응답 · LLM 심사 결과), FastAPI, LangGraph
checkpointer/store 기반 단기·장기 메모리.
