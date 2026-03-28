import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from retrieval import retrieve_docs

CHAT_MODEL = "models/gemini-2.5-flash"


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
    model_name = normalize_model_name(embedding_model)
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="literature",
    )


def format_context(docs):
    blocks = []
    sources = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = display_page(doc.metadata)
        excerpt = doc.page_content.strip().replace("\n", " ")
        blocks.append(f"[{i}] Source: {source}, page: {page}\nExcerpt: {excerpt}")
        sources.append((i, source, page))

    return "\n\n".join(blocks), sources


def fact_check_with_model(text: str, context: str):
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    prompt = f"""
You are a strict scientific fact-checking assistant.
Check the claim/paragraph only against the provided context from papers.

Task:
1) Classify as one of: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, INSUFFICIENT_EVIDENCE.
2) Give a short reason.
3) Provide evidence citations using [n] IDs from context.
4) If any part is wrong or unsupported, identify that exact part.

Output format exactly:
Verdict: <one label>
Confidence: <High/Medium/Low>
Reason:
- ...
Evidence:
- ... [n]
- ... [n]
Unsupported/Incorrect Parts:
- ...

Claim/Paragraph:
{text}

Context:
{context}
"""
    response = llm.invoke(prompt)
    return str(response.content)


def run_fact_check(db, text: str, top_k: int):
    docs = retrieve_docs(db, text, top_k=top_k)
    if not docs:
        print("No relevant chunks found.")
        return

    context, sources = format_context(docs)

    report = fact_check_with_model(text, context)

    print("\n=== Fact Check Report ===\n")
    print(report)
    print(f"\nModel used: {CHAT_MODEL}")

    print("\n=== Retrieved Sources ===\n")
    for idx, source, page in sources:
        print(f"[{idx}] {source} (page {page})")


def main():
    parser = argparse.ArgumentParser(
        description="Step 3 for RAG: Fact-check claims/paragraphs against uploaded papers."
    )
    parser.add_argument("--text", help="Claim or paragraph to fact-check.")
    parser.add_argument(
        "--persist-dir",
        default="vectorstore/chroma",
        help="Path to persisted Chroma DB.",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--embedding-model",
        default="models/gemini-embedding-001",
        help="Gemini embedding model used when querying vectors.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run step1_generate_embeddings.py first."
        )

    db = get_vectorstore(persist_dir, args.embedding_model)

    if args.text:
        run_fact_check(db, args.text, args.top_k)
        return

    print("Interactive fact-check mode.")
    print("Paste a claim/paragraph and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Fact-check> ").strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        run_fact_check(db, text, args.top_k)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
