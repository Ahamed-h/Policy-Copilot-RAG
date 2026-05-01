# Policy Copilot RAG

A retrieval-augmented assistant for answering employee policy, onboarding, and internal process questions with grounded context from internal PDF documents. The app uses LangChain Expression Language (LCEL) for streaming RAG inference, Chroma for persistent vector storage, BM25 plus dense retrieval for hybrid search, and Streamlit for the chat interface. [web:801][web:810]

## Overview

Policy Copilot indexes PDF documents from the `data_source/` folder and stores embeddings in a persistent Chroma collection so the knowledge base survives app restarts. Chroma supports persistent local storage through `persist_directory`, which makes it suitable for this style of local document QA workflow. [web:494][web:812]

The retrieval pipeline combines sparse keyword search and dense semantic retrieval, then reranks the candidate chunks before sending grounded context to the model. This style of LCEL pipeline supports composable chains and streaming output, which works well with Streamlit chat apps. [web:801][web:802][web:809]

The user interface is built with Streamlit, and answers are streamed token by token for a smoother chat experience. LCEL supports streaming and `RunnablePassthrough`, which is commonly used to pass the original question alongside retrieved context in RAG chains. [web:801][web:804]

## Project structure

```text
policy-Copilot-RAG/
├── build_index.py          # Reads PDFs, chunks them, adds metadata, builds Chroma index
├── rag.py                  # Hybrid retrieval, reranking, prompt, and inference chain
├── st_app.py               # Streamlit chat UI
├── data_source/            # Source policy PDFs
├── vector_stores/          # Persistent Chroma database
└── assets/                 # Images used in README/UI
```

## How it works

### 1. Document ingestion

`build_index.py` loads PDF files from `data_source/` using `PyPDFLoader`, extracts text, chunks it by section headings, and stores metadata such as source file, page, policy type, and section title. `PyPDFLoader` preserves page-level document metadata, which is useful for traceable answers and source display. [web:747][web:749]

### 2. Persistent indexing

The processed chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a persistent Chroma collection. Chroma can be recreated and loaded later using the same `collection_name`, embedding function, and `persist_directory`. [web:494][web:810]

### 3. Retrieval

The app uses hybrid retrieval:
- BM25 for exact keyword matching.
- Chroma vector retrieval for semantic similarity.
- An ensemble retriever to combine both signals.
- A cross-encoder reranker to select the best final chunks.

This setup helps the assistant answer both exact policy questions and semantically phrased employee queries more reliably. Hybrid retrieval and reranking are common improvements for higher-precision RAG systems. [web:783][web:707]

### 4. Answer generation

The final prompt is built with LCEL using a pipeline pattern like:

```python
chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
```

LCEL is designed for composable chains and supports streaming, parallel composition, and cleaner orchestration than older chain styles. [web:801][web:804][web:806]

## Features

- Grounded answers from uploaded internal policy PDFs. [web:747]
- Persistent Chroma vector database for local reuse across sessions. [web:810][web:812]
- Hybrid retrieval using BM25 and dense vector search. [web:783]
- Reranking with a cross-encoder for better final context quality. [web:707]
- Streamlit chat UI with token streaming. [web:801][web:809]
- Metadata-aware chunking for source file, section, and approximate page tracing. [web:747][web:749]
- Strict fallback behavior for missing information: the assistant can answer that the documents do not contain the requested detail.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rauni-iitr/langchain_chromaDB_opensourceLLM_streamlit.git
cd langchain_chromaDB_opensourceLLM_streamlit
```

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the core packages used by the current project:

```bash
pip install -U pypdf chromadb streamlit sentence-transformers transformers \
langchain langchain-community langchain-core langchain-classic \
langchain-google-genai
```

If you use the reranker in `rag.py`, ensure `sentence-transformers` is installed because the cross-encoder depends on it.

### 4. Set API key

The current `rag.py` uses Google Gemini through `langchain-google-genai`, so set one of these environment variables before running the app:

#### Windows PowerShell

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

#### Linux / macOS

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

## Build the index

Before running the app, place your policy PDFs inside `data_source/` and build the vector index:

```bash
python build_index.py
```

If you change chunking logic, metadata, or source documents, delete the `vector_stores/` folder and rebuild the index so Chroma does not keep stale chunks. Persistent Chroma collections reuse previously stored data until explicitly recreated. [web:494][web:812]

## Run the app

```bash
streamlit run st_app.py
```

The app will open in your browser and stream answers in a chat-style interface.

## Example questions

Use questions like these to test the pipeline:

- What are the types of leave available to employees?
- What is the policy for Paternity Leave?
- What is the standard probationary period for new hires?
- Are employees allowed to accept gifts from vendors?
- What happens if an employee is overpaid by mistake?

These questions help verify leave-policy retrieval, handbook retrieval, and conduct-policy retrieval separately.

## Notes

- `rag.py` uses LCEL because it provides a clean composition model and streaming-friendly inference flow. LCEL is commonly expressed as a pipe-style runnable chain. [web:801][web:804]
- `build_index.py` is responsible for ingestion quality; poor chunking will reduce answer quality even if the model is strong. Chunk quality is a major factor in RAG performance. [web:752][web:757]
- If you use Streamlit file upload instead of local PDFs later, PDF loaders often need a temporary file path rather than the in-memory upload object directly. [web:805][web:808]

## Screenshot

![Policy Copilot UI](./assets/snap1.png)