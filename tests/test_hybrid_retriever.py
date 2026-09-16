from agentflow.retrieval.bm25 import SimpleBM25Retriever
from agentflow.retrieval.hybrid import HybridRetriever
from agentflow.schemas import Evidence


def test_keyword_retriever_returns_relevant_document():
    docs = [
        Evidence(id="doc1__0", source="doc1.txt", text="LangGraph supports stateful agent workflows."),
        Evidence(id="doc2__0", source="doc2.txt", text="FastAPI is useful for web APIs."),
    ]
    keyword = SimpleBM25Retriever()
    keyword.add_documents(docs)
    retriever = HybridRetriever(keyword)

    results = retriever.search("stateful LangGraph agent", top_k=1)

    assert results
    assert results[0].id == "doc1__0"
