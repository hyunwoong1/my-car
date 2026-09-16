"""도메인 도구 정의."""
import os
import sqlite3
from datetime import date as _date
from typing import Optional

from langchain_core.tools import tool

from retriever import get_retriever


@tool
def search_vehicle_manual(query: str, vehicle_type: Optional[str] = None) -> str:
    """차량 매뉴얼(경고등 의미, 점검 주기, 자가정비 가능 항목, 고장 증상별 원인, 안전수칙 등)에서 관련 내용을 검색한다.
    질문에 특정 차종이 언급되면 vehicle_type에 sedan_1600(세단) 또는 suv_2000d(SUV) 중 해당하는 값을 지정한다."""
    retriever = get_retriever(vehicle_type=vehicle_type)
    docs = retriever.invoke(query)
    if not docs:
        return "관련 매뉴얼 내용을 찾지 못했습니다."
    return "\n\n".join(
        f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
        for d in docs
    )


# ── 정비이력 관리 (SQLite) ────────────────────────────────────────────

DB_PATH = "data/maintenance.db"


def _get_connection() -> sqlite3.Connection:
    """정비이력 SQLite DB 커넥션을 반환한다."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """차량 마스터·정비이력 테이블이 없으면 만들고, 비어 있으면 샘플 데이터를 시딩한다."""
    conn = _get_connection()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_no TEXT PRIMARY KEY,
            vehicle_type TEXT NOT NULL,
            registered_date TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS maintenance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_no TEXT NOT NULL REFERENCES vehicles(vehicle_no),
            date TEXT NOT NULL,
            item TEXT NOT NULL,
            cost INTEGER,
            next_due_date TEXT
        )"""
    )
    if conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO vehicles (vehicle_no, vehicle_type, registered_date) VALUES (?, ?, ?)",
            [
                ("12가3456", "sedan_1600", "2024-01-15"),
                ("34나5678", "suv_2000d", "2024-03-01"),
            ],
        )
        conn.executemany(
            "INSERT INTO maintenance_records (vehicle_no, date, item, cost, next_due_date) VALUES (?, ?, ?, ?, ?)",
            [
                ("12가3456", "2024-06-10", "엔진오일 교체", 80000, "2024-12-10"),
                ("12가3456", "2025-01-20", "타이어 위치 교환", 30000, "2025-07-20"),
                ("34나5678", "2024-09-05", "요소수 보충", 20000, None),
            ],
        )
    conn.commit()
    conn.close()


_init_db()


def _vehicle_exists(vehicle_no: str) -> bool:
    """차량 마스터에 해당 차량번호가 등록되어 있는지 확인한다."""
    conn = _get_connection()
    row = conn.execute("SELECT 1 FROM vehicles WHERE vehicle_no = ?", (vehicle_no,)).fetchone()
    conn.close()
    return row is not None


@tool
def register_vehicle(vehicle_no: str, vehicle_type: str) -> str:
    """차량을 등록한다. 차량번호와 차종(sedan_1600 또는 suv_2000d)을 입력받아 차량 마스터에 등록하며,
    이미 등록된 차량번호면 이미 등록되어 있다고 안내한다."""
    if _vehicle_exists(vehicle_no):
        return f"{vehicle_no} 차량은 이미 등록되어 있습니다."
    conn = _get_connection()
    conn.execute(
        "INSERT INTO vehicles (vehicle_no, vehicle_type, registered_date) VALUES (?, ?, ?)",
        (vehicle_no, vehicle_type, _date.today().isoformat()),
    )
    conn.commit()
    conn.close()
    return f"{vehicle_no}({vehicle_type}) 차량을 등록했습니다."


@tool
def list_vehicles() -> str:
    """등록된 차량 목록(차량번호, 차종, 등록일)을 조회한다."""
    conn = _get_connection()
    rows = conn.execute("SELECT vehicle_no, vehicle_type, registered_date FROM vehicles").fetchall()
    conn.close()
    if not rows:
        return "등록된 차량이 없습니다."
    return "\n".join(f"{r['vehicle_no']} / {r['vehicle_type']} / 등록일: {r['registered_date']}" for r in rows)


@tool
def get_maintenance_history(vehicle_no: str) -> str:
    """차량번호로 정비이력(날짜, 항목, 비용, 다음 점검 권장일)을 조회한다.
    등록되지 않은 차량번호면 지어내지 말고 등록되지 않았다고 안내한다."""
    if not _vehicle_exists(vehicle_no):
        return f"{vehicle_no}는 등록되지 않은 차량입니다."
    conn = _get_connection()
    rows = conn.execute(
        "SELECT id, date, item, cost, next_due_date FROM maintenance_records WHERE vehicle_no = ? ORDER BY date DESC",
        (vehicle_no,),
    ).fetchall()
    conn.close()
    if not rows:
        return f"{vehicle_no} 차량의 정비이력이 없습니다."
    return "\n".join(
        f"[{r['id']}] {r['date']} - {r['item']} (비용: {r['cost']}원, 다음 점검 권장일: {r['next_due_date']})"
        for r in rows
    )


@tool
def add_maintenance_record(
    vehicle_no: str,
    date: str,
    item: str,
    cost: Optional[int] = None,
    next_due_date: Optional[str] = None,
) -> str:
    """차량의 정비 내역을 새로 등록한다(날짜, 항목, 비용, 다음 점검 권장일).
    등록되지 않은 차량번호면 register_vehicle로 먼저 차량을 등록해야 한다고 안내한다."""
    if not _vehicle_exists(vehicle_no):
        return f"{vehicle_no}는 등록되지 않은 차량입니다. register_vehicle로 차량을 먼저 등록해주세요."
    conn = _get_connection()
    conn.execute(
        "INSERT INTO maintenance_records (vehicle_no, date, item, cost, next_due_date) VALUES (?, ?, ?, ?, ?)",
        (vehicle_no, date, item, cost, next_due_date),
    )
    conn.commit()
    conn.close()
    return f"{vehicle_no} 차량의 정비이력을 등록했습니다: {date} - {item}"


@tool
def update_maintenance_record(
    record_id: int,
    confirm: bool = False,
    date: Optional[str] = None,
    item: Optional[str] = None,
    cost: Optional[int] = None,
    next_due_date: Optional[str] = None,
) -> str:
    """정비이력(record_id)을 수정한다. 되돌리기 어려운 변경이므로 사용자에게 반드시 재확인을 받은 뒤
    confirm=True로 다시 호출해야 하며, confirm=False일 때는 절대 실행하지 않는다."""
    if not confirm:
        return "이 변경은 되돌리기 어렵습니다. 사용자에게 다시 한 번 확인한 뒤 confirm=True로 요청해주세요."
    conn = _get_connection()
    row = conn.execute("SELECT * FROM maintenance_records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        conn.close()
        return f"정비이력 {record_id}번을 찾을 수 없습니다."
    updates = {
        "date": date if date is not None else row["date"],
        "item": item if item is not None else row["item"],
        "cost": cost if cost is not None else row["cost"],
        "next_due_date": next_due_date if next_due_date is not None else row["next_due_date"],
    }
    conn.execute(
        "UPDATE maintenance_records SET date = ?, item = ?, cost = ?, next_due_date = ? WHERE id = ?",
        (updates["date"], updates["item"], updates["cost"], updates["next_due_date"], record_id),
    )
    conn.commit()
    conn.close()
    return f"정비이력 {record_id}번을 수정했습니다: {updates['date']} - {updates['item']}"


@tool
def delete_maintenance_record(record_id: int, confirm: bool = False) -> str:
    """정비이력(record_id)을 삭제한다. 되돌리기 어려운 변경이므로 사용자에게 반드시 재확인을 받은 뒤
    confirm=True로 다시 호출해야 하며, confirm=False일 때는 절대 실행하지 않는다."""
    if not confirm:
        return "이 삭제는 되돌릴 수 없습니다. 사용자에게 다시 한 번 확인한 뒤 confirm=True로 요청해주세요."
    conn = _get_connection()
    row = conn.execute("SELECT * FROM maintenance_records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        conn.close()
        return f"정비이력 {record_id}번을 찾을 수 없습니다."
    conn.execute("DELETE FROM maintenance_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return f"정비이력 {record_id}번을 삭제했습니다."
