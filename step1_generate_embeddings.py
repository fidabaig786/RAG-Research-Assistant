import argparse
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


LOW_SIGNAL_PATTERNS = [
    r"\breferences\b",
    r"\bbibliography\b",
    r"\backnowledg(e)?ments?\b",
    r"\bappendix\b",
    r"\ball rights reserved\b",
    r"\bcopyright\b",
    r"\bdoi\s*:",
]


def is_low_signal_chunk(text: str) -> bool:
    cleaned = " ".join(text.split())
    lowered = cleaned.lower()
    if len(cleaned) < 120:
        return True

    keyword_hits = sum(1 for p in LOW_SIGNAL_PATTERNS if re.search(p, lowered))
    alpha_ratio = sum(ch.isalpha() for ch in cleaned) / max(len(cleaned), 1)

    if keyword_hits >= 2:
        return True
    if alpha_ratio < 0.55:
        return True
    return False


def load_pdfs(data_dir: Path):
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = pdf.name
            d.metadata["source_path"] = str(pdf)
            raw_page = d.metadata.get("page")
            if isinstance(raw_page, int):
                d.metadata["page_index"] = raw_page
                d.metadata["page"] = raw_page + 1
        documents.extend(docs)

    return documents, pdf_files


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    raw_chunks = splitter.split_documents(documents)

    chunks = []
    for chunk in raw_chunks:
        if is_low_signal_chunk(chunk.page_content):
            continue
        chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def build_vectorstore(chunks, persist_dir: Path, embedding_model: str):
    model_name = embedding_model
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="literature",
    )
    return db


def main():
    parser = argparse.ArgumentParser(
        description="Step 1 for RAG: Generate embeddings from PDFs and persist to Chroma."
    )
    parser.add_argument(
        "--data-dir",
        default="Data",
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--persist-dir",
        default="vectorstore/chroma",
        help="Directory to store persisted vector DB.",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--embedding-model",
        default="models/gemini-embedding-001",
        help="Gemini embedding model name.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild existing vector store.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    data_dir = Path(args.data_dir)
    persist_dir = Path(args.persist_dir)

    if args.rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    documents, pdf_files = load_pdfs(data_dir)
    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    try:
        build_vectorstore(chunks, persist_dir, args.embedding_model)
    except Exception as e:
        raise RuntimeError("Embedding generation failed. Check API key, model access, and input PDFs.") from e

    print("Embedding generation completed")
    print(f"PDF files processed: {len(pdf_files)}")
    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks embedded: {len(chunks)}")
    print(f"Vector store path: {persist_dir.resolve()}")
    print(f"Embedding model: {args.embedding_model}")


if __name__ == "__main__":
    main()
