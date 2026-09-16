"""도메인 도구 정의."""
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
