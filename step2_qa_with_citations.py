import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from retrieval import retrieve_docs

CHAT_MODEL = "models/gemini-3.1-pro-preview"


def normalize_model_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"


def display_page(metadata: dict):
    page_index = metadata.get("page_index")
    if isinstance(page_index, int):
        return page_index + 1
    page = metadata.get("page", "unknown")
    if isinstance(page, int):
        return page + 1
    return page


def get_vectorstore(persist_dir: Path, embedding_model: str):
    embeddings = GoogleGenerativeAIEmbeddings(model=normalize_model_name(embedding_model))
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="literature",
    )


def format_context(docs):
    blocks = []
    source_index = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = display_page(doc.metadata)
        text = doc.page_content.strip().replace("\n", " ")
        blocks.append(f"[{i}] Source: {source}, page: {page}\nExcerpt: {text}")
        source_index.append((i, source, page))
    return "\n\n".join(blocks), source_index


def build_qa_prompt(question: str, context: str) -> str:
    return f"""
You are a literature assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say: "I don't have enough evidence in the retrieved papers."

Rules:
1) Cite evidence inline using bracket IDs like [1], [2].
2) Every key claim must have at least one citation.
3) Do not invent facts or references.

Question:
{question}

Context:
{context}
"""


def ask_model(prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    response = llm.invoke(prompt)
    return str(response.content)


def run_single_query(db, question: str, top_k: int, use_reranker: bool = True, reranker_strategy: str = "llm"):
    docs = retrieve_docs(db, question, top_k=top_k, use_reranker=use_reranker, reranker_strategy=reranker_strategy)
    if not docs:
        print("No relevant chunks found.")
        return

    context, source_index = format_context(docs)
    answer = ask_model(build_qa_prompt(question, context))

    print("\n=== Answer ===\n")
    print(answer)
    print(f"\nModel used: {CHAT_MODEL}")

    print("\n=== Sources ===\n")
    for idx, source, page in source_index:
        print(f"[{idx}] {source} (page {page})")


def main():
    parser = argparse.ArgumentParser(description="Step 2 for RAG: Ask questions with source citations.")
    parser.add_argument("--question", help="Question about the uploaded literature.")
    parser.add_argument("--persist-dir", default="vectorstore/chroma", help="Path to persisted Chroma DB.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--embedding-model",
        default="models/gemini-embedding-001",
        help="Gemini embedding model used when creating/querying vectors.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        default=True,
        help="Disable reranking with --no-reranker.",
    )
    parser.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Disable LLM-based reranking.",
    )
    parser.add_argument(
        "--reranker-strategy",
        default="llm",
        choices=["llm"],
        help="Reranking strategy (LLM-based).",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(f"Vector store not found at {persist_dir}. Run step1_generate_embeddings.py first.")

    db = get_vectorstore(persist_dir, args.embedding_model)

    if args.question:
        run_single_query(db, args.question, args.top_k, args.use_reranker, args.reranker_strategy)
        return

    print("Interactive mode. Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        run_single_query(db, question, args.top_k, args.use_reranker, args.reranker_strategy)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
