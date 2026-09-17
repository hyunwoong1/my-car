"""자체 평가 공통 유틸리티: 도구 호출 기록, LLM-as-Judge, RAGAS 스타일 지표 산출.

RAGAS(ragas) 패키지는 공용 .venv에 설치돼 있지 않고, mini-pjt만을 위해 여러 실습
폴더가 공유하는 .venv에 새 패키지를 넣는 건 영향 범위가 커서 피했다. 대신 RAGAS가
쓰는 것과 같은 개념(Faithfulness, Answer Relevancy, Context Relevance)을
day7_practice/llm_judge.py와 같은 방식의 LLM 심사로 직접 계산한다.
"""
import csv
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

load_dotenv()

JUDGE_MODEL = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
judge_llm = ChatBedrockConverse(model=JUDGE_MODEL, region_name="us-east-1", temperature=0)


def get_text(message) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환한다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


class ToolRecorder(BaseCallbackHandler):
    """Supervisor 구조에서는 서브 에이전트의 도구 호출이 최상위 메시지에 안 남으므로,
    콜백으로 실제 호출된 도구/거쳐간 에이전트/도구 결과(컨텍스트)를 전부 기록한다."""

    def __init__(self):
        self.tools: list[str] = []
        self.agents: list[str] = []
        self.contexts: list[str] = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "")
        if name.startswith("transfer_to_"):
            self.agents.append(name[len("transfer_to_"):])
        elif name and not name.startswith("transfer_back"):
            self.tools.append(name)

    def on_tool_end(self, output, **kwargs):
        text = str(getattr(output, "content", output))
        if not text.startswith("Successfully transferred") and not text.startswith("Transferring"):
            self.contexts.append(text[:1000])


def load_test_queries(path: str = "evaluation/test_queries.csv") -> list[dict]:
    """세미콜론으로 구분된 expected_traits/forbidden/expected_tools를 리스트로 분해해 로드한다."""
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("expected_traits", "forbidden", "expected_tools"):
            value = row.get(key, "") or ""
            row[key] = [v.strip() for v in value.split(";") if v.strip() and v.strip() != "-"]
    return rows


class CaseJudgment(BaseModel):
    """RAGAS가 쓰는 개념(Faithfulness/Answer Relevancy/Context Relevance)을 LLM 심사로 직접 계산한다."""

    traits_met: list[bool] = Field(description="expected_traits 각 항목이 답변에 충족됐는지, 순서대로")
    forbidden_triggered: list[bool] = Field(description="forbidden 각 항목이 답변에서 실제로 발생했는지, 순서대로")
    faithfulness: float = Field(ge=0, le=1, description="답변이 도구 호출 결과(컨텍스트)에 근거하고 지어내지 않은 정도")
    answer_relevancy: float = Field(ge=0, le=1, description="답변이 질문의 요구에 얼마나 적절히 대응하는지")
    context_relevance: float = Field(ge=0, le=1, description="도구 호출 결과(컨텍스트)가 질문과 관련 있는 정도")
    reasoning: str = Field(description="판정 근거를 한국어 1~3문장으로")


judge = judge_llm.with_structured_output(CaseJudgment)

JUDGE_PROMPT = """당신은 차량 관리 에이전트의 답변을 심사하는 평가자입니다.
아래 정보를 보고 각 항목을 판정하세요.

[질문]
{question}

[에이전트 최종 답변]
{answer}

[호출된 도구]
{tools}

[도구 호출 결과(컨텍스트, 최대 몇 개만 발췌)]
{contexts}

[답변에 반드시 있어야 할 특성 목록 — 각각 충족했는지 순서대로 true/false]
{traits}

[답변에 절대 있으면 안 되는 것 목록 — 각각 실제로 발생했는지 순서대로 true/false]
{forbidden}

faithfulness(답변이 위 컨텍스트/도구 결과에 근거하고 지어내지 않았는가),
answer_relevancy(질문 요구에 얼마나 적절히 답했는가),
context_relevance(도구 호출 결과가 질문과 관련 있는가, 도구를 안 썼다면 1.0)도 0~1 사이로 매기세요."""


def judge_case(question: str, answer: str, tools: list[str], contexts: list[str],
                traits: list[str], forbidden: list[str]) -> CaseJudgment:
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer or "(빈 답변)",
        tools=", ".join(tools) if tools else "(없음)",
        contexts="\n---\n".join(contexts[:5]) if contexts else "(없음)",
        traits="\n".join(f"{i+1}. {t}" for i, t in enumerate(traits)) if traits else "(없음)",
        forbidden="\n".join(f"{i+1}. {f}" for i, f in enumerate(forbidden)) if forbidden else "(없음)",
    )
    for attempt in range(3):
        try:
            return judge.invoke(prompt)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def run_case(app, row: dict) -> dict:
    """케이스 하나를 실제로 실행하고 심사까지 마쳐 결과 dict를 반환한다."""
    from langchain_core.messages import HumanMessage

    rec = ToolRecorder()
    result = app.invoke(
        {"messages": [HumanMessage(content=row["input"])]},
        {"callbacks": [rec], "recursion_limit": 25},
    )
    answer = get_text(result["messages"][-1])
    verdict = judge_case(row["input"], answer, rec.tools, rec.contexts, row["expected_traits"], row["forbidden"])

    traits_ok = all(verdict.traits_met) if row["expected_traits"] else True
    forbidden_ok = not any(verdict.forbidden_triggered) if row["forbidden"] else True
    passed = traits_ok and forbidden_ok

    return {
        "id": row["id"],
        "category": row["category"],
        "input": row["input"],
        "answer": answer,
        "tools_called": rec.tools,
        "agents_visited": rec.agents,
        "expected_tools": row["expected_tools"],
        "expected_traits": row["expected_traits"],
        "traits_met": verdict.traits_met,
        "forbidden": row["forbidden"],
        "forbidden_triggered": verdict.forbidden_triggered,
        "faithfulness": verdict.faithfulness,
        "answer_relevancy": verdict.answer_relevancy,
        "context_relevance": verdict.context_relevance,
        "reasoning": verdict.reasoning,
        "passed": passed,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    passed = sum(r["passed"] for r in results)
    by_category: dict[str, list[int]] = {}
    for r in results:
        by_category.setdefault(r["category"], [0, 0])
        by_category[r["category"]][1] += 1
        by_category[r["category"]][0] += int(r["passed"])
    avg = lambda key: round(sum(r[key] for r in results) / n, 3) if n else 0.0
    return {
        "total": n,
        "passed": passed,
        "pass_rate": round(passed / n, 3) if n else 0.0,
        "by_category": by_category,
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_relevance": avg("context_relevance"),
    }


def write_report(path: str, round_no: int, results: list[dict], summary: dict, prev_summary: dict | None = None) -> None:
    lines = [f"# 자체 평가 리포트 — Round {round_no}", ""]
    lines.append(f"- 실행 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 전체 통과: {summary['passed']}/{summary['total']} ({summary['pass_rate']*100:.0f}%)")
    lines.append(f"- Faithfulness 평균: {summary['faithfulness']}")
    lines.append(f"- Answer Relevancy 평균: {summary['answer_relevancy']}")
    lines.append(f"- Context Relevance 평균: {summary['context_relevance']}")
    lines.append("")

    if prev_summary is not None:
        lines.append("## Round {} 대비 개선폭".format(round_no - 1))
        d_pass = summary["passed"] - prev_summary["passed"]
        lines.append(
            f"- 통과: {prev_summary['passed']}/{prev_summary['total']} → "
            f"{summary['passed']}/{summary['total']} ({'+' if d_pass >= 0 else ''}{d_pass}건)"
        )
        for metric in ("faithfulness", "answer_relevancy", "context_relevance"):
            d = round(summary[metric] - prev_summary[metric], 3)
            lines.append(f"- {metric}: {prev_summary[metric]} → {summary[metric]} ({'+' if d >= 0 else ''}{d})")
        lines.append("")
        lines.append("### 카테고리별 변화")
        for cat in sorted(set(summary["by_category"]) | set(prev_summary["by_category"])):
            p_now, n_now = summary["by_category"].get(cat, [0, 0])
            p_prev, n_prev = prev_summary["by_category"].get(cat, [0, 0])
            lines.append(f"- {cat}: {p_prev}/{n_prev} → {p_now}/{n_now}")
        lines.append("")

    lines.append("## 카테고리별 통과")
    for cat, (p, n) in summary["by_category"].items():
        lines.append(f"- {cat}: {p}/{n}")
    lines.append("")

    lines.append("## 케이스별 상세")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(f"### [{mark}] {r['id']} ({r['category']})")
        lines.append(f"- 질문: {r['input']}")
        lines.append(f"- 답변: {r['answer'][:400]}{'...' if len(r['answer']) > 400 else ''}")
        lines.append(f"- 호출된 도구: {r['tools_called']} / 거친 에이전트: {r['agents_visited']}")
        if r["expected_traits"]:
            trait_lines = [f"{'✅' if ok else '❌'} {t}" for t, ok in zip(r["expected_traits"], r["traits_met"])]
            lines.append(f"- 특성 충족: {'; '.join(trait_lines)}")
        if r["forbidden"]:
            forbid_lines = [f"{'🚫발생' if bad else '✅없음'} {f}" for f, bad in zip(r["forbidden"], r["forbidden_triggered"])]
            lines.append(f"- 금지 항목: {'; '.join(forbid_lines)}")
        lines.append(
            f"- Faithfulness: {r['faithfulness']} / Answer Relevancy: {r['answer_relevancy']} / "
            f"Context Relevance: {r['context_relevance']}"
        )
        lines.append(f"- 판정 근거: {r['reasoning']}")
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
