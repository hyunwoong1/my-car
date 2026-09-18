"""실행 기록을 JSONL 파일로 남기는 콜백 핸들러."""
import json
import time
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler


def get_text(message):
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환한다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


class FileTracer(BaseCallbackHandler):
    """LLM 호출, 도구 호출을 JSONL 한 줄씩 기록한다."""

    def __init__(self, path: str = "trace.jsonl"):
        self.path = Path(path)
        self._starts = {}  # run_id -> 시작 시각

    def _write(self, record: dict):
        record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- LLM ----
    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._starts[run_id] = time.time()
        flat = [get_text(m)[:200] for batch in messages for m in batch]
        self._write({"event": "llm_start", "run_id": str(run_id), "input": flat})

    def on_llm_end(self, response, *, run_id, **kw):
        elapsed = time.time() - self._starts.pop(run_id, time.time())
        usage = {}
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg is not None and getattr(msg, "usage_metadata", None):
            usage = msg.usage_metadata  # 입력, 출력 토큰 수
        self._write({
            "event": "llm_end", "run_id": str(run_id),
            "output": gen.text[:200], "latency_s": round(elapsed, 2),
            "usage": usage,
        })

    # ---- 도구 ----
    def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        self._starts[run_id] = time.time()
        self._write({"event": "tool_start", "run_id": str(run_id),
                     "tool": serialized.get("name"), "input": input_str[:200]})

    def on_tool_end(self, output, *, run_id, **kw):
        elapsed = time.time() - self._starts.pop(run_id, time.time())
        self._write({"event": "tool_end", "run_id": str(run_id),
                     "output": str(output)[:200], "latency_s": round(elapsed, 2)})

    def on_llm_error(self, error, *, run_id, **kw):
        self._write({"event": "llm_error", "run_id": str(run_id), "error": str(error)})

    def log_error(self, error: Exception) -> None:
        """LLM 콜백 밖에서 잡힌 예외(API 핸들러의 try/except 등)를 기록한다. 사용자에게는
        일반적인 오류 메시지만 보여주고, 실제 예외 내용은 이 파일에만 남기기 위한 용도다."""
        self._write({"event": "server_error", "error": str(error)})


class RequestRecorder(BaseCallbackHandler):
    """API 요청 한 건 동안 거친 에이전트·도구와 도구 결과(컨텍스트)를 순서대로 모은다.
    FileTracer(trace.jsonl 전체 로그)와 달리, 이 요청 하나에 대한 answer/contexts/trace
    구조화 응답을 만드는 데 쓴다."""

    def __init__(self):
        self.trace: list[dict] = []
        self.contexts: list[str] = []
        self._names: dict = {}  # run_id -> 도구 이름 (on_tool_end에서 핸드오프 여부 판단용)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name", "")
        self._names[run_id] = name
        if name.startswith("transfer_to_"):
            self.trace.append({"type": "agent", "name": name[len("transfer_to_"):]})
        elif name and not name.startswith("transfer_back"):
            self.trace.append({"type": "tool", "name": name, "args": input_str})

    def on_tool_end(self, output, *, run_id, **kwargs):
        # 핸드오프 도구(transfer_to_*/transfer_back_to_*)는 실제 검색 결과가 아니라 그래프
        # 라우팅용 Command 객체를 반환하므로, 텍스트가 아니라 도구 이름으로 걸러내야 한다.
        # (Command에는 .content가 없어 str(output) 결과가 그대로 흘러들어와 있었다.)
        name = self._names.pop(run_id, "")
        if name.startswith("transfer_to_") or name.startswith("transfer_back"):
            return
        self.contexts.append(str(getattr(output, "content", output)))
