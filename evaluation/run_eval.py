"""evaluation/run_eval.py - 인-아웃 세트 자체 평가 실행.

사용법: python evaluation/run_eval.py <round_no>
  round_no=1이면 evaluation/round1_report.md만 만든다.
  round_no=2 이상이면 이전 라운드의 summary json을 읽어 개선폭도 함께 리포트에 적는다.
"""
import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "evaluation")

from eval_lib import load_test_queries, run_case, summarize, write_report  # noqa: E402


def main(round_no: int) -> None:
    from agent import app  # mini-pjt의 Supervisor 그래프

    cases = load_test_queries()
    results = [run_case(app, row) for row in cases]
    summary = summarize(results)

    prev_summary = None
    if round_no > 1:
        try:
            with open(f"evaluation/round{round_no - 1}_summary.json", encoding="utf-8") as f:
                prev_summary = json.load(f)
        except FileNotFoundError:
            pass

    write_report(f"evaluation/round{round_no}_report.md", round_no, results, summary, prev_summary)
    with open(f"evaluation/round{round_no}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(f"evaluation/round{round_no}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Round {round_no}: {summary['passed']}/{summary['total']} 통과")
    print(f"faithfulness={summary['faithfulness']} answer_relevancy={summary['answer_relevancy']} "
          f"context_relevance={summary['context_relevance']}")


if __name__ == "__main__":
    round_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(round_arg)
