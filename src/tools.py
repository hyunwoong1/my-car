"""도메인 도구 정의."""
import json
import os
from datetime import date as _date
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import Column, Integer, String, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

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
SEED_PATH = "data/seed.json"

# 스키마 정의부터 CRUD 도구까지 전부 SQLAlchemy ORM으로 처리한다(단일 소스, 원시 SQL 없음).
Base = declarative_base()


class _Vehicle(Base):
    __tablename__ = "vehicles"
    vehicle_no = Column(String, primary_key=True)
    vehicle_type = Column(String, nullable=False)
    registered_date = Column(String, nullable=False)


class _MaintenanceRecord(Base):
    __tablename__ = "maintenance_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_no = Column(String, nullable=False)
    date = Column(String, nullable=False)
    item = Column(String, nullable=False)
    cost = Column(Integer)
    next_due_date = Column(String)


_engine = create_engine(f"sqlite:///{DB_PATH}")
_Session = sessionmaker(bind=_engine)


def _seed_if_empty() -> None:
    """차량 마스터가 비어 있을 때만 data/seed.json의 샘플 데이터를 읽어 ORM으로 시딩한다."""
    session = _Session()
    if session.query(func.count(_Vehicle.vehicle_no)).scalar() == 0:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            seed = json.load(f)
        session.add_all(_Vehicle(**row) for row in seed["vehicles"])
        session.add_all(_MaintenanceRecord(**row) for row in seed["maintenance_records"])
        session.commit()
    session.close()


def _get_session() -> Session:
    """정비이력 DB에 대한 ORM 세션을 반환한다."""
    return _Session()


def _init_db() -> None:
    """ORM 모델 스키마로 테이블을 만들고, 비어 있으면 data/seed.json을 ORM으로 시딩한다."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(_engine, checkfirst=True)
    _seed_if_empty()


_init_db()


def _vehicle_exists(vehicle_no: str) -> bool:
    """차량 마스터에 해당 차량번호가 등록되어 있는지 확인한다."""
    session = _get_session()
    exists = session.query(_Vehicle).filter_by(vehicle_no=vehicle_no).first() is not None
    session.close()
    return exists


@tool
def register_vehicle(vehicle_no: str, vehicle_type: str) -> str:
    """차량을 등록한다. 차량번호와 차종(sedan_1600 또는 suv_2000d)을 입력받아 차량 마스터에 등록하며,
    이미 등록된 차량번호면 이미 등록되어 있다고 안내한다."""
    if _vehicle_exists(vehicle_no):
        return f"{vehicle_no} 차량은 이미 등록되어 있습니다."
    session = _get_session()
    session.add(
        _Vehicle(
            vehicle_no=vehicle_no,
            vehicle_type=vehicle_type,
            registered_date=_date.today().isoformat(),
        )
    )
    session.commit()
    session.close()
    return f"{vehicle_no}({vehicle_type}) 차량을 등록했습니다."


@tool
def list_vehicles() -> str:
    """등록된 차량 목록(차량번호, 차종, 등록일)을 조회한다."""
    session = _get_session()
    rows = session.query(_Vehicle).all()
    session.close()
    if not rows:
        return "등록된 차량이 없습니다."
    return "\n".join(f"{r.vehicle_no} / {r.vehicle_type} / 등록일: {r.registered_date}" for r in rows)


@tool
def get_maintenance_history(vehicle_no: str) -> str:
    """차량번호로 정비이력(날짜, 항목, 비용, 다음 점검 권장일)을 조회한다. 결과에 차종(vehicle_type)도
    함께 포함되므로, 이어서 매뉴얼을 검색할 때는 그 차종을 참고해야 한다.
    등록되지 않은 차량번호면 지어내지 말고 등록되지 않았다고 안내한다."""
    session = _get_session()
    vehicle = session.query(_Vehicle).filter_by(vehicle_no=vehicle_no).first()
    if vehicle is None:
        session.close()
        return f"{vehicle_no}는 등록되지 않은 차량입니다."
    rows = (
        session.query(_MaintenanceRecord)
        .filter_by(vehicle_no=vehicle_no)
        .order_by(_MaintenanceRecord.date.desc())
        .all()
    )
    header = f"차종: {vehicle.vehicle_type}"
    if not rows:
        session.close()
        return f"{header}\n{vehicle_no} 차량의 정비이력이 없습니다."
    body = "\n".join(
        f"[{r.id}] {r.date} - {r.item} (비용: {r.cost}원, 다음 점검 권장일: {r.next_due_date})"
        for r in rows
    )
    session.close()
    return f"{header}\n{body}"


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
    session = _get_session()
    session.add(
        _MaintenanceRecord(
            vehicle_no=vehicle_no, date=date, item=item, cost=cost, next_due_date=next_due_date
        )
    )
    session.commit()
    session.close()
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
    session = _get_session()
    row = session.get(_MaintenanceRecord, record_id)
    if row is None:
        session.close()
        return f"정비이력 {record_id}번을 찾을 수 없습니다."
    if date is not None:
        row.date = date
    if item is not None:
        row.item = item
    if cost is not None:
        row.cost = cost
    if next_due_date is not None:
        row.next_due_date = next_due_date
    session.commit()
    result = f"정비이력 {record_id}번을 수정했습니다: {row.date} - {row.item}"
    session.close()
    return result


@tool
def delete_maintenance_record(record_id: int, confirm: bool = False) -> str:
    """정비이력(record_id)을 삭제한다. 되돌리기 어려운 변경이므로 사용자에게 반드시 재확인을 받은 뒤
    confirm=True로 다시 호출해야 하며, confirm=False일 때는 절대 실행하지 않는다."""
    if not confirm:
        return "이 삭제는 되돌릴 수 없습니다. 사용자에게 다시 한 번 확인한 뒤 confirm=True로 요청해주세요."
    session = _get_session()
    row = session.get(_MaintenanceRecord, record_id)
    if row is None:
        session.close()
        return f"정비이력 {record_id}번을 찾을 수 없습니다."
    session.delete(row)
    session.commit()
    session.close()
    return f"정비이력 {record_id}번을 삭제했습니다."


# ── 지역 정보 검색 (더미, 이후 네이버 지역검색 API로 교체 예정) ──────────

PLACES_PATH = "data/local_places.json"


def _load_places() -> list[dict]:
    """지역 장소 더미 데이터를 읽어 반환한다."""
    import json

    with open(PLACES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _search_naver_local(query: str, display: int = 5) -> list[dict]:
    """네이버 지역검색 API 호출을 대신하는 더미 구현.
    실제 API로 교체할 때는 이 함수 내부만 requests.get(...) 호출로 바꾸면 된다."""
    keywords = query.lower().split()
    places = _load_places()

    def score(place: dict) -> int:
        haystack = " ".join(
            str(place.get(field, "")) for field in ("title", "category", "address", "description")
        ).lower()
        return sum(1 for kw in keywords if kw in haystack)

    ranked = [p for p in places if score(p) > 0]
    ranked.sort(key=score, reverse=True)
    return ranked[:display]


@tool
def search_local_places(query: str, display: int = 5) -> str:
    """지역/업종 검색어로 장소 정보(이름, 카테고리, 주소, 전화번호 등)를 조회하는 지역 검색 도구다.
    이 도구는 정비소 전용이 아닌 범용 지역 검색이므로, 정비소를 찾으려면 검색어에 지역명과 함께
    '카센터' 또는 '자동차정비' 같은 업종 키워드를 반드시 함께 넣어야 한다."""
    results = _search_naver_local(query, display=display)
    if not results:
        return "검색 결과가 없습니다."
    return "\n\n".join(
        f"[{p['title']}] {p['category']}\n"
        f"주소: {p['address']}\n"
        f"전화: {p['telephone']}"
        for p in results
    )
