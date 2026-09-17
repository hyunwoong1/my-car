"""evaluation/run_eval.py - 인-아웃 세트 자체 평가 실행.

사용법: python evaluation/run_eval.py <round_no>
  round_no=1이면 evaluation/round1_report.md만 만든다(최초 1회성 기록, 이후 덮어쓰지 않음).
  round_no=2(기본값)는 이후 반복 개선 사이클마다 계속 덮어써서
  evaluation/round2_report.md 하나에 최신 상태만 남긴다.

정비이력 DB는 evaluation/maintenance_test.db라는 별도 테스트 DB를 쓴다(운영용
data/maintenance.db는 건드리지 않음). 매 실행 전 이 테스트 DB 파일을 지우고
data/seed.json 기준으로 다시 시딩해, q03(등록) 케이스가 반복 실행 때마다
데이터를 누적시켜 q04/q08 같은 다른 케이스를 오염시키는 문제를 막는다.

Supervisor의 단기/장기 기억(체크포인트·스토어)도 이제 SQLite 파일로 영속화되므로, 같은 이유로
evaluation/checkpoints_test.sqlite · evaluation/store_test.sqlite라는 별도 테스트 파일을 쓰고
매 실행 전 지운다. 그렇지 않으면 케이스마다 고정된 thread_id/user_id(f"eval-{id}")에 이전
라운드의 대화 기록·기억한 차종이 남아, q09처럼 "차종이 불명확하면 되묻는지"를 확인하는 케이스가
과거 실행 때 기억해둔 차종 때문에 더 이상 되묻지 않는 식으로 오염될 수 있다.
"""
import json
import os
import sys

TEST_DB_PATH = "evaluation/maintenance_test.db"
TEST_CHECKPOINT_PATH = "evaluation/checkpoints_test.sqlite"
TEST_STORE_PATH = "evaluation/store_test.sqlite"
for path in (TEST_DB_PATH, TEST_CHECKPOINT_PATH, TEST_STORE_PATH):
    if os.path.exists(path):
        os.remove(path)
os.environ["MAINTENANCE_DB_PATH"] = TEST_DB_PATH
os.environ["CHECKPOINT_DB_PATH"] = TEST_CHECKPOINT_PATH
os.environ["STORE_DB_PATH"] = TEST_STORE_PATH

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
    round_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(round_arg)
