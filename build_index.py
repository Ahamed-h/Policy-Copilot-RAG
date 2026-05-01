import os
import re
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER_PATH = os.path.join(BASE_DIR, "data_source")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vector_stores")
COLLECTION_NAME = "policy_copilot"


def detect_policy_type(filename: str):
    f = filename.lower()
    if "leave" in f:
        return "leave"
    if "conduct" in f or "expectations" in f:
        return "conduct"
    if "handbook" in f:
        return "handbook"
    return "general"


def clean_text(text: str):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_heading(text: str):
    text = clean_text(text)

    heading_pattern = r"(?=^\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,100}$)"
    parts = re.split(heading_pattern, text, flags=re.MULTILINE)

    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(r"^\d+(?:\.\d+)*\.?\s+", part):
            if current.strip():
                chunks.append(current.strip())
            current = part
        else:
            current = f"{current}\n{part}".strip() if current else part

    if current.strip():
        chunks.append(current.strip())

    final_chunks = []
    for chunk in chunks:
        words = chunk.split()

        if len(words) < 20:
            if final_chunks:
                final_chunks[-1] = final_chunks[-1] + "\n" + chunk
            else:
                final_chunks.append(chunk)
        else:
            final_chunks.append(chunk)

    return final_chunks if final_chunks else [text]


def extract_section_id_and_title(chunk: str):
    first_line = chunk.split("\n")[0].strip()
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$", first_line)
    if m:
        return m.group(1), m.group(2).strip()
    return "unknown", first_line[:120]


def approximate_page(chunk: str, page_records: list):
    probe = re.sub(r"\s+", " ", chunk[:160]).strip()
    if not probe:
        return "unknown"

    for record in page_records:
        page_text_norm = re.sub(r"\s+", " ", record["text"])
        if probe[:80] in page_text_norm:
            return record["page"]

    return page_records[0]["page"] if page_records else "unknown"


def build_index():
    if not os.path.exists(PDF_FOLDER_PATH):
        raise FileNotFoundError(f"Folder not found: {PDF_FOLDER_PATH}")

    docs = []

    for file in os.listdir(PDF_FOLDER_PATH):
        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER_PATH, file)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        policy_type = detect_policy_type(file)

        page_records = []
        full_text_parts = []

        for page_doc in pages:
            page_text = clean_text(page_doc.page_content)
            page_num = page_doc.metadata.get("page", "unknown")

            if page_text:
                page_records.append({
                    "page": page_num,
                    "text": page_text
                })
                full_text_parts.append(page_text)

        full_text = "\n\n".join(full_text_parts)
        full_text = clean_text(full_text)

        if not full_text:
            continue

        chunks = split_by_heading(full_text)

        for i, chunk in enumerate(chunks):
            section_id, section_title = extract_section_id_and_title(chunk)
            approx_page = approximate_page(chunk, page_records)

            enriched_chunk = (
                f"Document: {file}\n"
                f"Policy Type: {policy_type}\n"
                f"Section: {section_title}\n\n"
                f"{chunk}"
            )

            docs.append(
                Document(
                    page_content=enriched_chunk,
                    metadata={
                        "source_file": file,
                        "page": approx_page,
                        "policy_type": policy_type,
                        "section_id": section_id,
                        "section_title": section_title,
                        "chunk_id": i
                    }
                )
            )

    if not docs:
        raise ValueError("No chunks created from PDFs.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )

    print(f"Indexed {len(docs)} chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()