// 정비노트 — 채팅 UI. 실제 백엔드(POST /query/stream, POST /query)와 통신한다.
(() => {
  "use strict";

  const DEFAULT_API_BASE = "http://localhost:8000";
  const getApiBase = () => localStorage.getItem("apiBase") || DEFAULT_API_BASE;
  const setApiBase = (value) => localStorage.setItem("apiBase", value);

  // 같은 thread_id를 계속 보내야 서버(SqliteSaver 체크포인터)가 같은 대화로 이어받는다.
  // 안 보내면 서버가 매 요청을 새 대화로 취급해 문맥이 끊긴다 — "새 대화 시작"을 누를 때만 새로 발급한다.
  function generateThreadId() {
    return typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
  let threadId = generateThreadId();

  // ---------- Chat ----------
  const messagesEl = document.getElementById("chatMessages");
  const formEl = document.getElementById("chatForm");
  const inputEl = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const statusEl = document.getElementById("connStatus");

  function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = `msg msg--${role}`;
    const p = document.createElement("p");
    p.textContent = text;
    div.appendChild(p);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function setStatus(kind, label) {
    statusEl.textContent = label;
    statusEl.className = `status status--${kind}`;
  }

  function setBusy(isBusy) {
    inputEl.disabled = isBusy;
    sendBtn.disabled = isBusy;
    sendBtn.dataset.loading = String(isBusy);
  }

  async function streamQuery(question, onToken) {
    const res = await fetch(`${getApiBase()}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: threadId }),
    });
    if (!res.ok || !res.body) {
      throw new Error(`서버가 ${res.status}로 응답했습니다.`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;
    let errorMessage = null;

    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const lines = rawEvent.split("\n");
        const eventLine = lines.find((l) => l.startsWith("event: "));
        const dataLine = lines.find((l) => l.startsWith("data: "));
        if (!eventLine || !dataLine) continue;

        const eventName = eventLine.slice("event: ".length).trim();
        const payload = JSON.parse(dataLine.slice("data: ".length));

        if (eventName === "token") {
          onToken(payload.text);
        } else if (eventName === "done") {
          finished = true;
        } else if (eventName === "error") {
          errorMessage = payload.message;
          finished = true;
        }
      }
    }
    if (errorMessage) throw new Error(errorMessage);
  }

  async function handleSubmit(question) {
    addMessage("user", question);
    setBusy(true);
    setStatus("idle", "연결 중");

    const agentBubble = addMessage("agent", "");
    agentBubble.classList.add("is-typing");
    const agentText = agentBubble.querySelector("p");
    let received = "";

    try {
      await streamQuery(question, (chunk) => {
        received += chunk;
        agentText.textContent = received;
        agentBubble.classList.remove("is-typing");
        messagesEl.scrollTop = messagesEl.scrollHeight;
      });
      if (!received) {
        agentText.textContent = "답변을 받지 못했습니다. 다시 시도해 주세요.";
      }
      agentBubble.classList.remove("is-typing");
      setStatus("ok", "200 OK");
    } catch (err) {
      agentBubble.classList.remove("is-typing");
      agentText.textContent =
        `연결에 실패했습니다: ${err.message} — 서버(${getApiBase()})가 실행 중인지, ` +
        "CORS가 허용되어 있는지 확인해 주세요.";
      setStatus("error", "오류");
    } finally {
      setBusy(false);
      inputEl.focus();
    }
  }

  formEl.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = inputEl.value.trim();
    if (!question) return;
    inputEl.value = "";
    handleSubmit(question);
  });

  function startNewChat() {
    threadId = generateThreadId(); // 새 thread_id를 발급해야 서버도 진짜 새 대화로 취급한다
    messagesEl.innerHTML = "";
    addMessage(
      "agent",
      "안녕하세요. 차량 매뉴얼, 정비이력, 근처 정비소 관련 질문을 도와드릴 수 있습니다."
    );
    setStatus("idle", "대기");
    inputEl.focus();
  }

  // ---------- Rail (side nav — new chat, example prompts, mobile drawer) ----------
  const rail = document.getElementById("rail");
  const railToggle = document.getElementById("railToggle");
  const railBackdrop = document.getElementById("railBackdrop");
  const newChatBtn = document.getElementById("newChatBtn");

  function openRail() {
    rail.classList.add("is-open");
    railToggle.setAttribute("aria-expanded", "true");
    railBackdrop.hidden = false;
  }
  function closeRail() {
    rail.classList.remove("is-open");
    railToggle.setAttribute("aria-expanded", "false");
    railBackdrop.hidden = true;
  }

  railToggle.addEventListener("click", () => {
    rail.classList.contains("is-open") ? closeRail() : openRail();
  });
  railBackdrop.addEventListener("click", closeRail);
  newChatBtn.addEventListener("click", () => {
    startNewChat();
    closeRail();
  });
  document.querySelectorAll(".rail__item[data-example]").forEach((btn) => {
    btn.addEventListener("click", () => {
      closeRail();
      runExample(btn.dataset.example);
    });
  });

  // ---------- Info dialog (동작 방식 · FAQ) ----------
  const infoDialog = document.getElementById("infoDialog");
  const infoBtn = document.getElementById("infoBtn");
  let infoLastFocused = null;

  function openInfo() {
    infoLastFocused = document.activeElement;
    closeRail();
    infoDialog.hidden = false;
    document.querySelector(".shell").inert = true;
  }
  function closeInfo() {
    infoDialog.hidden = true;
    document.querySelector(".shell").inert = false;
    if (infoLastFocused) infoLastFocused.focus();
  }
  infoBtn.addEventListener("click", openInfo);
  infoDialog.querySelectorAll("[data-close-info]").forEach((el) => {
    el.addEventListener("click", closeInfo);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !infoDialog.hidden) closeInfo();
  });

  // ---------- Command palette (⌘K) ----------
  const cmdk = document.getElementById("cmdk");
  const cmdkInput = document.getElementById("cmdkInput");
  const cmdkList = document.getElementById("cmdkList");
  const cmdkTrigger = document.getElementById("cmdkTrigger");
  const cmdkTriggerMobile = document.getElementById("cmdkTriggerMobile");
  const apiConfigBtn = document.getElementById("apiConfigBtn");

  const COMMANDS = [
    {
      label: "예시 질문 — 타이어 공기압 점검 주기",
      hint: "sedan_1600",
      run: () => runExample("세단 타이어 공기압은 얼마나 자주 점검해야 해?"),
    },
    {
      label: "예시 질문 — 브레이크 소음 진단",
      hint: "manual",
      run: () => runExample("브레이크 밟을 때 끼익 소리가 나는데 어떻게 해야 해?"),
    },
    {
      label: "예시 질문 — 근처 정비소 찾기",
      hint: "location",
      run: () => runExample("강남역 근처 정비소 좀 찾아줘"),
    },
    {
      label: "새 대화 시작",
      hint: "reset",
      run: startNewChat,
    },
    {
      label: "API 주소 설정",
      hint: getApiBase(),
      run: () => {
        const next = prompt("백엔드 주소를 입력하세요 (예: http://localhost:8000)", getApiBase());
        if (next) setApiBase(next.trim());
      },
    },
  ];

  function runExample(question) {
    document.getElementById("chat")?.scrollIntoView({ behavior: "smooth", block: "center" });
    inputEl.value = question;
    handleSubmit(question);
    inputEl.value = "";
  }

  let selectedIndex = 0;
  let filtered = COMMANDS;
  let lastFocused = null;

  function renderCmdkList() {
    cmdkList.innerHTML = "";
    filtered.forEach((cmd, i) => {
      const li = document.createElement("li");
      li.role = "option";
      li.setAttribute("aria-selected", String(i === selectedIndex));
      const label = document.createElement("span");
      label.textContent = cmd.label;
      const hint = document.createElement("span");
      hint.className = "hint";
      hint.textContent = cmd.hint;
      li.append(label, hint);
      li.addEventListener("click", () => runCommand(cmd));
      cmdkList.appendChild(li);
    });
  }

  function runCommand(cmd) {
    closeCmdk();
    cmd.run();
  }

  function openCmdk() {
    lastFocused = document.activeElement;
    closeRail();
    cmdk.hidden = false;
    document.querySelector(".shell").inert = true;
    cmdkInput.value = "";
    selectedIndex = 0;
    filtered = COMMANDS;
    renderCmdkList();
    cmdkInput.focus();
  }

  function closeCmdk() {
    cmdk.hidden = true;
    document.querySelector(".shell").inert = false;
    if (lastFocused) lastFocused.focus();
  }

  cmdkTrigger.addEventListener("click", openCmdk);
  cmdkTriggerMobile.addEventListener("click", openCmdk);
  cmdk.querySelector("[data-close]").addEventListener("click", closeCmdk);
  apiConfigBtn.addEventListener("click", () => COMMANDS.find((c) => c.hint.includes("://") || c.label.includes("API")).run());

  cmdkInput.addEventListener("input", () => {
    const q = cmdkInput.value.toLowerCase();
    filtered = COMMANDS.filter((c) => c.label.toLowerCase().includes(q));
    selectedIndex = 0;
    renderCmdkList();
  });

  cmdkInput.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, filtered.length - 1);
      renderCmdkList();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
      renderCmdkList();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (filtered[selectedIndex]) runCommand(filtered[selectedIndex]);
    } else if (e.key === "Escape") {
      closeCmdk();
    }
  });

  document.addEventListener("keydown", (e) => {
    const isK = e.key.toLowerCase() === "k";
    if ((e.metaKey || e.ctrlKey) && isK) {
      e.preventDefault();
      cmdk.hidden ? openCmdk() : closeCmdk();
    } else if (e.key === "Escape" && !cmdk.hidden) {
      closeCmdk();
    }
  });
})();
