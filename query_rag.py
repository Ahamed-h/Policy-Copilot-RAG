import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vector_stores")
COLLECTION_NAME = "policy_copilot"

_embeddings = None
_vectordb = None
_docs_cache = None
_hybrid_retriever = None
_reranker = None
_llm = None
_chain = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

def get_vectordb():
    global _vectordb
    if _vectordb is None:
        _vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME
        )
    return _vectordb

def load_all_docs():
    global _docs_cache
    if _docs_cache is None:
        raw = get_vectordb().get(include=["documents", "metadatas"])
        docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(raw.get("documents", []), raw.get("metadatas", []))]
        _docs_cache = docs
    return _docs_cache

def get_hybrid_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        docs = load_all_docs()
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = 12
        dense = get_vectordb().as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 24})
        _hybrid_retriever = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
    return _hybrid_retriever

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank_docs(query: str, docs, top_n=5):
    if not docs: return []
    scores = get_reranker().predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]

def detect_policy_filter(query: str):
    q = query.lower()
    if any(term in q for term in ["leave", "casual", "sick", "maternity", "paternity"]): return "leave"
    if any(term in q for term in ["conduct", "ethics", "misconduct", "harassment"]): return "conduct"
    if any(term in q for term in ["probation", "onboarding", "handbook"]): return "handbook"
    return None

def retrieve_and_format(query: str):
    # 1. Hybrid Retrieval
    docs = get_hybrid_retriever().invoke(query)
    
    # 2. Metadata Filtering
    policy_filter = detect_policy_filter(query)
    if policy_filter:
        docs = [d for d in docs if d.metadata.get("policy_type") == policy_filter] or docs
    
    # 3. Reranking
    top_docs = rerank_docs(query, docs, top_n=5)
    
    if not top_docs: return "No relevant context retrieved."
    
    return "\n\n".join([f"Source: {d.metadata.get('source_file')}\n{d.page_content}" for d in top_docs])

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, streaming=True)
    return _llm

def get_chain():
    global _chain
    if _chain is None:
        prompt = PromptTemplate.from_template("""
You are Policy Copilot. Answer based ONLY on the context.
Rules:
- Concise, decision-oriented.
- If answer not in context, say: "I do not know based on the provided documents."
- List items clearly.
- If values like "number of days" are missing (template), say so.

Context: {context}
Question: {question}
Answer:""")
        _chain = ({"context": retrieve_and_format, "question": RunnablePassthrough()} | prompt | get_llm() | StrOutputParser())
    return _chain

def inference(query: str):
    return get_chain().stream(query)