"""차량 매뉴얼 문서를 읽어 청크로 나누고 Chroma 벡터스토어에 적재·검색하는 RAG 파이프라인.
벡터 검색과 BM25 키워드 검색을 EnsembleRetriever로 묶은 하이브리드 검색을 쓴다."""
import os
import re
import threading
import time
from glob import glob
from typing import Optional

from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
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


_chunks: Optional[list[Document]] = None


def _load_chunks() -> list[Document]:
    """매뉴얼 문서를 글자 수 기준(중복 포함)으로 분할해 청크 목록을 반환한다(프로세스당 한 번만 계산)."""
    global _chunks
    if _chunks is None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        _chunks = splitter.split_documents(_load_documents())
    return _chunks


_vectorstore: Optional[Chroma] = None
_vectorstore_lock = threading.Lock()


def _build_vectorstore() -> Chroma:
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


def get_vectorstore() -> Chroma:
    """Chroma 벡터스토어를 반환한다. 프로세스당 한 번만 만들어 재사용한다(같은 경로로 PersistentClient를
    여러 번 새로 열면 chromadb가 이전 클라이언트를 정리하는 과정에서 오류를 내는 경우가 있어 캐싱한다).
    LangGraph의 ToolNode는 도구를 별도 워커 스레드에서 실행하므로, 락으로 동시 생성 시도를 막고
    chromadb의 Rust 바인딩이 간헐적으로 내는 초기화 오류(AttributeError/ValueError)는 재시도로 넘긴다."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _vectorstore_lock:
        if _vectorstore is None:
            last_error = None
            for attempt in range(3):
                try:
                    _vectorstore = _build_vectorstore()
                    break
                except (AttributeError, ValueError) as e:
                    last_error = e
                    time.sleep(0.2)
            else:
                raise last_error
    return _vectorstore


_kiwi = Kiwi()
_KEEP_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "SL", "SN"}


def _korean_tokenizer(text: str) -> list[str]:
    """한국어 형태소 단위 토크나이저. 명사, 어간, 숫자, 외래어만 남긴다."""
    return [t.form for t in _kiwi.tokenize(text) if t.tag in _KEEP_TAGS]


_bm25_lock = threading.Lock()
_bm25_cache: dict[Optional[str], BM25Retriever] = {}


def _get_bm25(vehicle_type: Optional[str], k: int) -> BM25Retriever:
    """차종별로 BM25 검색기를 만들어 캐싱한다(벡터스토어처럼 vehicle_type으로 문서를 좁힌다)."""
    with _bm25_lock:
        bm25 = _bm25_cache.get(vehicle_type)
        if bm25 is None:
            chunks = _load_chunks()
            if vehicle_type:
                chunks = [c for c in chunks if c.metadata.get("vehicle_type") == vehicle_type]
            bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_korean_tokenizer)
            _bm25_cache[vehicle_type] = bm25
        bm25.k = k
        return bm25


def get_supported_vehicle_types() -> list[str]:
    """매뉴얼 파일(data/manuals/*.md)의 파일명에서 실제 검색 가능한 차종 목록을 뽑는다.
    새 차종 매뉴얼 파일을 추가하기만 하면 이 목록에도 자동으로 반영된다."""
    return sorted(
        os.path.splitext(os.path.basename(path))[0]
        for path in glob(os.path.join(MANUALS_DIR, "*.md"))
    )


def get_retriever(vehicle_type: Optional[str] = None, k: int = 3):
    """벡터 검색과 BM25 키워드 검색을 묶은 하이브리드 리트리버를 만든다.
    vehicle_type이 주어지면 두 검색기 모두 해당 차종 문서로만 검색을 제한한다."""
    search_kwargs = {"k": k}
    if vehicle_type:
        search_kwargs["filter"] = {"vehicle_type": vehicle_type}
    vector = get_vectorstore().as_retriever(search_kwargs=search_kwargs)
    bm25 = _get_bm25(vehicle_type, k)
    return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.3, 0.7])
