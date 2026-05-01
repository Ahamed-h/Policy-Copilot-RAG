import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vector_stores")
COLLECTION_NAME = "policy_copilot"

# --- Initialization ---
_embeddings = None
_vectordb = None
_hybrid_retriever = None

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

def get_hybrid_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        raw = get_vectordb().get(include=["documents", "metadatas"])
        docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(raw.get("documents", []), raw.get("metadatas", []))]
        
        bm25 = BM25Retriever.from_documents(docs); bm25.k = 12
        dense = get_vectordb().as_retriever(search_kwargs={"k": 12})
        _hybrid_retriever = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
    return _hybrid_retriever

# --- Retrieval Logic ---
def rerank_docs(query, docs, top_n=5):
    if not docs: return []
    scores = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict([(query, d.page_content) for d in docs])
    return [d for d, s in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]]

def retrieve_and_format(query: str):
    # 1. Hybrid Retrieval
    docs = get_hybrid_retriever().invoke(query)
    
    # 2. Domain Detection for Inclusive Filtering
    q = query.lower()
    domains = {"leave": ["leave", "paternity"], "conduct": ["conduct", "gift"], "handbook": ["probation", "onboarding"]}
    active_filters = [d for d, kws in domains.items() if any(k in q for k in kws)]
    
    # Apply filter only if single domain detected, otherwise return all
    if len(active_filters) == 1:
        docs = [d for d in docs if d.metadata.get("policy_type") == active_filters[0]] or docs
        
    # 3. Reranking for Precision
    top_docs = rerank_docs(query, docs, top_n=8)
    return "\n\n".join([f"Source: {d.metadata.get('source_file')}\n{d.page_content}" for d in top_docs])

# --- Chain ---
import os
import streamlit as st


def get_chain():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "CRITICAL: API Key not found. Please set 'GOOGLE_API_KEY' "
            "in your environment variables or in Streamlit Secrets."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
You are Policy Copilot. Answer based ONLY on the context.
Rules:
- Be concise and grounded.
- If the answer is not in the context, say: "I do not know based on the provided documents."
- List items clearly when useful.
- If the source text contains blanks like "_____", say the information is missing in the document.
- Do not say the user sent an empty message unless the question is actually empty.

Context:
{context}

Question:
{question}

Answer:
""")

    return (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
def inference(query: str): return get_chain().stream(query)