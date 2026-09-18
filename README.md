# 미니 PJT: 개인 차량 관리 Agent

## 무엇을 푸나
경고등·이상 증상 대처법을 모르고 여러 차량의 정비이력도 기억하기 어려운 문제를, 매뉴얼 지식과 내 차량 데이터로 바로 답해주는 개인용 차량 관리 에이전트로 해결한다.

## 활용한 패턴 (Day 1~7)
- Day 1: 구조화 출력 — API 응답(`QueryResponse`/`TraceStep`)과 평가 심사 결과(`CaseJudgment`)를 Pydantic 구조화 출력으로 반환
- Day 2: RAG 하이브리드 검색 — BM25+벡터 `EnsembleRetriever` + `vehicle_type` 메타데이터 필터(`retriever.py`)로 다른 차종 매뉴얼이 안 섞이게 함
- Day 3: ReAct(도구 자율 선택) — 서브 에이전트(`create_agent`)가 매뉴얼 검색·CRUD·지역 검색 도구를 자율 선택·호출
- Day 4: 도구 다중 결합 — 정비이력 CRUD(SQLite)와 지역 검색 도구를 한 질의에서 필요 시 순차 결합(`tools.py`)
- Day 6: Multi-Agent Supervisor — `create_supervisor`로 매뉴얼 검색·정비이력·정비소 조회 3개 서브 에이전트를 역할 분할·위임(`agent.py`), 이 프로젝트의 핵심 패턴
- Day 7: Observability + 평가 — 자체 콜백 기반 `FileTracer`로 LLM·도구 호출을 `trace.jsonl`에 기록(`tracer.py`), LLM-as-Judge로 평가 지표를 산출(`eval_lib.py`)

## 아키텍처

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

Supervisor가 질문을 분류해 3개 서브 에이전트로 위임하고, 여러 에이전트가 필요하면 순서대로 호출해
하나로 종합한다. 단기 기억(`SqliteSaver`)만 두고 장기 기억은 두지 않는다(다차량 사용자 오적용 방지).
`web/`는 이 API에 붙는 정적 채팅 UI다. 자세한 내용은 [SERVICE.md](SERVICE.md) 참고.

## 실행 방법

```bash
cd mini-pjt

# 콘솔 데모 (질의응답 스트리밍 확인)
python src/agent.py

# 자체 평가 (evaluation/maintenance_test.db를 매번 초기화 후 실행)
python evaluation/run_eval.py
```

`.env`(레포 루트)에 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_DEFAULT_REGION` 필요(Amazon
Bedrock). 에이전트/심사 모델은 기본값이 `claude-sonnet-4-6`이며 `.env`의 `AGENT_MODEL`/`JUDGE_MODEL`로
바꿀 수 있다.

### 1) API 서버 띄우기

`src/` 모듈들이 flat import라 `PYTHONPATH`를 `src`로 잡아준 뒤 `mini-pjt` 루트에서 uvicorn으로 띄운다.

```bash
cd mini-pjt
$env:PYTHONPATH = "src"; uvicorn api_server:api --port 8000
```

curl로 바로 확인해볼 수도 있다:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "12가3456 차량 마지막 정비가 언제였어?", "thread_id": "demo-1"}'
```

웹 채팅 화면 열기: API 서버가 떠 있는 상태에서 `web/index.html`을 브라우저로 열면 된다(빌드 불필요).

## RAGAS 평가 결과
RAGAS 라이브러리 대신 같은 개념을 LLM-as-Judge로 직접 계산했다(`eval_lib.py`).

- faithfulness: 0.978
- answer_relevancy: 0.926
- context_relevance: 0.882 (RAGAS의 context_precision과 같은 개념으로 자체 계산한 대체 지표)

## 인-아웃 세트 통과율 (자체 평가)
- 1차 (Day 9 종료): 7 / 12 통과
- 2차 (Day 10 개선 후): 12 / 12 통과
- 개선폭: +5건 

## 트라이앤에러 회고
- 시도했지만 실패한 접근 · 왜 실패했는가
  - 매뉴얼 청킹을 헤더 기준으로 나누려 했으나, 지금은 더미 md 파일이라도 실제로는 PDF/HTML 같은
    비정형 문서가 들어올 수 있어 포맷에 의존하는 헤더 분할은 위험 — `RecursiveCharacterTextSplitter`(글자
    수 기준)로 전환했다(`retriever.py`).
  - 차종 정보를 세션을 넘는 장기 기억으로 남겨 재확인을 줄이려 했으나, 한 사용자가 차종이 다른
    차량을 여러 대 가질 수 있어 이전에 확인한 차종을 다른 차량 질문에 잘못 넘겨짚을 위험이 있어
    제거했다(커밋 `ebc17c0`). 같은 대화(`thread_id`) 안에서만 문맥을 이어받는다.
  - 핸드오프 도구(`transfer_to_*`) 결과가 실제 검색 결과인 것처럼 평가 컨텍스트에 섞여 들어간 적이
    있었다 — `Command` 객체엔 `.content`가 없어 `str(output)`이 그대로 들어간 게 원인이었고, 텍스트
    내용이 아니라 도구 이름으로 핸드오프 여부를 걸러내도록 고쳤다(`tracer.py`).
- 최종 채택한 접근 · 왜 그것으로 갔는가
  - `langgraph_supervisor.create_supervisor`로 매뉴얼 검색·정비이력·정비소 조회를 역할별 서브
    에이전트로 분리 — 각 에이전트가 자기 담당이 아닌 질문에는 답을 지어내지 않게 프롬프트로 통제하기
    쉬워서.
  - BM25+벡터 하이브리드 검색 + `vehicle_type` 메타데이터 필터 — 키워드(BM25)와 의미(벡터) 검색을
    함께 써야 "브레이크 소리" 같은 증상 표현과 매뉴얼 문구가 다를 때도 놓치지 않는다.
  - `SqliteSaver` 단기 기억만 두고 장기 기억은 의도적으로 비움 — 다차량 사용자에게 안전한 쪽을
    택함.
- 남은 한계 · 향후 개선 방향
  - 정비소 조회는 아직 `data/local_places.json` 더미 기반 — 네이버 지역검색 API로 교체 예정(함수
    시그니처는 이미 대비되어 있음, `tools.py`의 `_search_naver_local`).
  - 웹 UI는 새로고침·재접속해도 `thread_id`를 localStorage에 유지해 서버 쪽 대화 문맥은 이어지지만,
    화면에 보이던 말풍선(transcript) 자체는 복원되지 않는다 — 별도 대화이력 저장소가 필요해 계획만
    세운 상태.

## 핵심 코드 위치
- `src/agent.py:80` — Supervisor 그래프 생성(`create_supervisor`), `src/agent.py:98` — 체크포인터와 함께 컴파일
- `src/tools.py:15` — RAG 검색 도구(`search_vehicle_manual`), `src/tools.py:106` 이하 — 차량/정비이력 CRUD 도구
- `src/retriever.py:141` — BM25+벡터 하이브리드 리트리버 생성(`get_retriever`)
- `src/api_server.py:51`, `src/api_server.py:83` — `POST /query`, `POST /query/stream` 엔드포인트
- `evaluation/eval_lib.py:99` — LLM-as-Judge 평가 로직(`run_case`)
- `web/` — API 연동 웹 채팅 UI
