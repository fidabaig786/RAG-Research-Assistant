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


def parse_outline(outline: str):
    items = [line.strip(" -\t") for line in outline.splitlines() if line.strip()]
    return items if items else [outline.strip()]


def retrieve_for_outline(db, outline_items, top_k_per_item: int):
    collected = []
    seen = set()

    for item in outline_items:
        docs = retrieve_docs(db, item, top_k=top_k_per_item)
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "na")
            key = (source, page, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            collected.append(doc)

    return collected


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


def generate_review(llm, outline: str, context: str):
    prompt = f"""
You are an academic writing assistant for literature reviews.
Write a concise, structured literature review using ONLY the provided context.

Rules:
1) Follow the user's outline/themes exactly.
2) Add inline citations using [n] from context for each key claim.
3) Do not invent facts or papers.
4) If evidence is weak for any section, explicitly say so.
5) Keep an academic tone.

User outline/themes:
{outline}

Context:
{context}
"""
    response = llm.invoke(prompt)
    return response.content


def run_review(db, outline: str, top_k_per_item: int):
    outline_items = parse_outline(outline)
    docs = retrieve_for_outline(db, outline_items, top_k_per_item)

    if not docs:
        print("No relevant chunks found for the provided outline.")
        return

    context, sources = format_context(docs)

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
    review = generate_review(llm, outline, context)

    print("\n=== Literature Review Draft ===\n")
    print(review)
    print(f"\nModel used: {CHAT_MODEL}")

    print("\n=== Retrieved Sources ===\n")
    for idx, source, page in sources:
        print(f"[{idx}] {source} (page {page})")


def main():
    parser = argparse.ArgumentParser(
        description="Step 4 for RAG: Generate literature review draft from outline/themes."
    )
    parser.add_argument(
        "--outline",
        help="Outline or thematic points (multi-line string supported).",
    )
    parser.add_argument(
        "--persist-dir",
        default="vectorstore/chroma",
        help="Path to persisted Chroma DB.",
    )
    parser.add_argument("--top-k-per-item", type=int, default=4)
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

    if args.outline:
        run_review(db, args.outline, args.top_k_per_item)
        return

    print("Interactive literature review mode.")
    print("Enter your outline/themes (single line), then press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        outline = input("Outline> ").strip()
        if not outline:
            continue
        if outline.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        run_review(db, outline, args.top_k_per_item)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
