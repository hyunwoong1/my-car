"""정비소 조회 에이전트 그래프: 범용 지역 검색 도구를 이용해 정비소를 찾는 단일 에이전트."""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from tools import search_local_places
from tracer import get_text

load_dotenv()

llm = ChatBedrockConverse(
    model=os.environ.get("AGENT_MODEL") or "us.anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
    temperature=0,
)

location_agent = create_agent(
    llm,
    [search_local_places],
    system_prompt=(
        "당신은 근처 정비소를 찾아주는 도우미입니다. "
        "search_local_places 도구는 정비소 전용이 아닌 범용 지역 검색 도구이므로, "
        "정비소를 찾을 때는 사용자가 말한 지역명에 '카센터' 또는 '자동차정비' 같은 "
        "업종 키워드를 조합한 검색어로 호출하세요. "
        "검색 결과가 없으면 지어내지 말고 없다고 안내하세요."
    ),
    name="location_agent",
)


if __name__ == "__main__":
    question = "강남역 근처에 차 정비할 곳 있어?"
    result = location_agent.invoke({"messages": [HumanMessage(question)]})
    print(get_text(result["messages"][-1]))
