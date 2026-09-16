"""차량 매뉴얼 문서를 읽어 청크로 나누고 Chroma 벡터스토어에 적재·검색하는 RAG 파이프라인."""
import os
import re
from glob import glob
from typing import Optional

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MANUALS_DIR = "data/manuals"
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "vehicle_manuals"

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """파일 앞부분의 YAML 프런트매터(--- ... ---)를 제거한다."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def _load_documents() -> list[Document]:
    """data/manuals 아래 매뉴얼 md 파일들을 읽어 Document 목록으로 반환한다."""
    documents = []
    for path in sorted(glob(os.path.join(MANUALS_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        vehicle_type = os.path.splitext(os.path.basename(path))[0]
        documents.append(
            Document(
                page_content=_strip_frontmatter(raw),
                metadata={"vehicle_type": vehicle_type, "source": os.path.basename(path)},
            )
        )
    return documents


def _load_chunks() -> list[Document]:
    """매뉴얼 문서를 글자 수 기준(중복 포함)으로 분할해 청크 목록을 반환한다."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(_load_documents())


def get_vectorstore() -> Chroma:
    """Chroma 벡터스토어를 반환한다. 없으면 새로 빌드하고, 있으면 그대로 불러와 재사용한다."""
    if os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    return Chroma.from_documents(
        documents=_load_chunks(),
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )


def get_retriever(vehicle_type: Optional[str] = None, k: int = 3):
    """벡터스토어에서 리트리버를 만든다. vehicle_type이 주어지면 해당 차종 문서로만 검색을 제한한다."""
    search_kwargs = {"k": k}
    if vehicle_type:
        search_kwargs["filter"] = {"vehicle_type": vehicle_type}
    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)
