import argparse
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval import retrieve_docs

CHAT_MODEL = "models/gemini-2.5-flash"

LOW_SIGNAL_PATTERNS = (
    r"\breferences\b",
    r"\bbibliography\b",
    r"\backnowledg(e)?ments?\b",
    r"\bappendix\b",
    r"\ball rights reserved\b",
    r"\bcopyright\b",
    r"\bdoi\s*:",
)


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


def load_pdfs(data_dir: Path):
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    documents = []
    for pdf in pdf_files:
        docs = PyPDFLoader(str(pdf)).load()
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    raw_chunks = splitter.split_documents(documents)
    chunks = [c for c in raw_chunks if not is_low_signal_chunk(c.page_content)]
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def generate_embeddings(data_dir: Path, persist_dir: Path, embedding_model: str, chunk_size: int, chunk_overlap: int, rebuild: bool):
    if rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents, pdf_files = load_pdfs(data_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)

    embeddings = GoogleGenerativeAIEmbeddings(model=normalize_model_name(embedding_model))
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="literature",
    )

    print("✅ Embeddings generated")
    print(f"PDF files: {len(pdf_files)} | Pages: {len(documents)} | Chunks: {len(chunks)}")


def get_vectorstore(persist_dir: Path, embedding_model: str):
    embeddings = GoogleGenerativeAIEmbeddings(model=normalize_model_name(embedding_model))
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="literature",
    )


def format_context(docs):
    blocks, sources = [], []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = display_page(doc.metadata)
        text = doc.page_content.strip().replace("\n", " ")
        blocks.append(f"[{i}] Source: {source}, page: {page}\nExcerpt: {text}")
        sources.append((i, source, page))
    return "\n\n".join(blocks), sources


def ask_model(prompt: str, chat_model: str = CHAT_MODEL):
    llm = ChatGoogleGenerativeAI(model=normalize_model_name(chat_model), temperature=0)
    response = llm.invoke(prompt)
    return response.content, normalize_model_name(chat_model)


def print_sources(sources):
    print("\n=== Sources ===\n")
    for i, s, p in sources:
        print(f"[{i}] {s} (page {p})")


def enrich_citations_in_text(text: str, sources):
    source_map = {i: (s, p) for i, s, p in sources}

    def replace_single(match):
        idx = int(match.group(1))
        if idx not in source_map:
            return match.group(0)
        src, page = source_map[idx]
        return f"[{src}, page {page}]"

    def replace_group(match):
        raw = match.group(1)
        ids = [x.strip() for x in re.split(r"[,;]", raw) if x.strip().isdigit()]
        if not ids:
            return match.group(0)
        parts = []
        for item in ids:
            idx = int(item)
            if idx in source_map:
                src, page = source_map[idx]
                parts.append(f"{src}, page {page}")
            else:
                parts.append(item)
        return "[" + "; ".join(parts) + "]"

    # First handle grouped citations like [1, 2] or [1;2] (with optional spaces)
    text = re.sub(r"\[\s*((?:\d+\s*[,;]\s*)+\d+)\s*\]", replace_group, text)
    # Then handle single citations like [3] or [ 3 ]
    text = re.sub(r"\[\s*(\d+)\s*\]", replace_single, text)
    return text


def append_citation_legend(text: str, sources):
    source_map = {i: (s, p) for i, s, p in sources}
    cited_ids = sorted({int(x) for x in re.findall(r"\[(\d+)(?::[^\]]*)?\]", text)})
    if not cited_ids:
        return text

    lines = ["", "Citations used in this answer:"]
    for idx in cited_ids:
        if idx in source_map:
            src, page = source_map[idx]
            lines.append(f"- [{idx}] {src} (page {page})")
    if len(lines) == 2:
        return text
    return text.rstrip() + "\n" + "\n".join(lines)


def run_qa(db, question: str, top_k: int, chat_model: str = CHAT_MODEL):
    docs = retrieve_docs(db, question, top_k=top_k)
    if not docs:
        print("No relevant chunks found.")
        return

    context, sources = format_context(docs)
    prompt = f"""
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
    answer, used_model = ask_model(prompt, chat_model)
    answer = enrich_citations_in_text(answer, sources)
    answer = append_citation_legend(answer, sources)
    print("\n=== Answer ===\n")
    print(answer)
    print(f"\nModel used: {used_model}")
    print_sources(sources)


def run_fact_check(db, text: str, top_k: int, chat_model: str = CHAT_MODEL):
    docs = retrieve_docs(db, text, top_k=top_k)
    if not docs:
        print("No relevant chunks found.")
        return

    context, sources = format_context(docs)
    prompt = f"""
You are a strict scientific fact-checking assistant.
Check the claim/paragraph only against the provided context.

Task:
1) Classify as: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, INSUFFICIENT_EVIDENCE.
2) Give short reason.
3) Cite evidence with [n].
4) Identify unsupported parts.

Output format:
Verdict: <label>
Confidence: <High/Medium/Low>
Reason:
- ...
Evidence:
- ... [n]
Unsupported/Incorrect Parts:
- ...

Claim/Paragraph:
{text}

Context:
{context}
"""
    report, used_model = ask_model(prompt, chat_model)
    report = enrich_citations_in_text(report, sources)
    report = append_citation_legend(report, sources)
    print("\n=== Fact Check Report ===\n")
    print(report)
    print(f"\nModel used: {used_model}")
    print_sources(sources)


def parse_outline(outline: str):
    items = [line.strip(" -\t") for line in outline.splitlines() if line.strip()]
    return items if items else [outline.strip()]


def run_lit_review(db, outline: str, top_k_per_item: int, chat_model: str = CHAT_MODEL):
    outline_items = parse_outline(outline)

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

    if not collected:
        print("No relevant chunks found.")
        return

    context, sources = format_context(collected)
    prompt = f"""
You are an academic writing assistant.
Write a concise literature review using ONLY provided context.

Rules:
1) Follow the user outline/themes.
2) Add inline citations [n] for each key claim.
3) Do not invent facts/papers.
4) If evidence is weak, state that clearly.

Outline/Themes:
{outline}

Context:
{context}
"""
    review, used_model = ask_model(prompt, chat_model)
    review = enrich_citations_in_text(review, sources)
    review = append_citation_legend(review, sources)
    print("\n=== Literature Review Draft ===\n")
    print(review)
    print(f"\nModel used: {used_model}")
    print_sources(sources)


def run_all_in_one(
    db,
    question: str | None = None,
    claim_text: str | None = None,
    outline: str | None = None,
    qa_top_k: int = 4,
    fact_top_k: int = 6,
    review_top_k_per_item: int = 4,
    chat_model: str = CHAT_MODEL,
):
    ran_any = False

    if question and question.strip():
        run_qa(db, question.strip(), top_k=qa_top_k, chat_model=chat_model)
        ran_any = True

    if claim_text and claim_text.strip():
        run_fact_check(db, claim_text.strip(), top_k=fact_top_k, chat_model=chat_model)
        ran_any = True

    if outline and outline.strip():
        run_lit_review(
            db,
            outline.strip(),
            top_k_per_item=review_top_k_per_item,
            chat_model=chat_model,
        )
        ran_any = True

    if not ran_any:
        print("Nothing to run. Provide at least one of: question, claim text, or outline.")


def extract_after_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    lowered = text.lower().strip()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()


def parse_tagged_all_in_one(user_input: str):
    # Supported format in one line:
    # q: ... ; claim: ... ; review: ...
    pattern = re.compile(r"(q|question|claim|fact|text|review|outline)\s*:\s*", re.IGNORECASE)
    matches = list(pattern.finditer(user_input))
    if not matches:
        return None, None, None

    values = {}
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(user_input)
        values[key] = user_input[start:end].strip(" ;\t\n")

    question = values.get("q") or values.get("question")
    claim_text = values.get("claim") or values.get("fact") or values.get("text")
    outline = values.get("review") or values.get("outline")
    return question, claim_text, outline


def route_prompt(db, user_input: str, chat_model: str = CHAT_MODEL):
    text = user_input.strip()
    if not text:
        print("Please enter a prompt.")
        return

    q, claim_text, outline = parse_tagged_all_in_one(text)
    if q or claim_text or outline:
        run_all_in_one(
            db,
            question=q,
            claim_text=claim_text,
            outline=outline,
            qa_top_k=4,
            fact_top_k=6,
            review_top_k_per_item=4,
            chat_model=chat_model,
        )
        return

    lowered = text.lower()
    fact_keywords = (
        "verify",
        "fact-check",
        "fact check",
        "is this true",
        "check this claim",
        "validate this",
        "claim:",
    )
    review_keywords = (
        "literature review",
        "write a review",
        "draft a review",
        "review:",
        "outline:",
        "summarize literature",
    )

    if any(k in lowered for k in fact_keywords):
        claim_text = extract_after_prefix(text, ("verify:", "fact:", "claim:", "fact-check:"))
        run_fact_check(db, claim_text, top_k=6, chat_model=chat_model)
        return

    if any(k in lowered for k in review_keywords):
        outline = extract_after_prefix(text, ("review:", "outline:", "write:"))
        run_lit_review(db, outline, top_k_per_item=4, chat_model=chat_model)
        return

    run_qa(db, text, top_k=4, chat_model=chat_model)


def interactive_mode(db, chat_model: str = CHAT_MODEL):
    print("Unified RAG CLI (single prompt mode)")
    print("Type one prompt. The CLI auto-detects: question, fact-check, or review.")
    print("Use 'exit' to quit.")
    print("Optional one-line all-in-one format: q: ... ; claim: ... ; review: ...")

    while True:
        user_input = input("YOU> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        route_prompt(db, user_input, chat_model=chat_model)


def main():
    parser = argparse.ArgumentParser(description="Unified CLI for embedding, Q&A, fact-checking, and literature review.")
    parser.add_argument("--data-dir", default="Data")
    parser.add_argument("--persist-dir", default="vectorstore/chroma")
    parser.add_argument("--embedding-model", default="models/gemini-embedding-001")

    subparsers = parser.add_subparsers(dest="command")

    p_embed = subparsers.add_parser("embed", help="Generate embeddings")
    p_embed.add_argument("--chunk-size", type=int, default=800)
    p_embed.add_argument("--chunk-overlap", type=int, default=100)
    p_embed.add_argument("--rebuild", action="store_true")

    p_ask = subparsers.add_parser("ask", help="Q&A with citations")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--top-k", type=int, default=4)

    p_fact = subparsers.add_parser("fact", help="Fact-check claim/paragraph")
    p_fact.add_argument("--text", required=True)
    p_fact.add_argument("--top-k", type=int, default=6)

    p_review = subparsers.add_parser("review", help="Generate literature review draft")
    p_review.add_argument("--outline", required=True)
    p_review.add_argument("--top-k-per-item", type=int, default=4)

    p_all = subparsers.add_parser("all", help="Run Q&A, fact-check, and review in one command")
    p_all.add_argument("--question", default="")
    p_all.add_argument("--text", default="")
    p_all.add_argument("--outline", default="")
    p_all.add_argument("--top-k", type=int, default=4)
    p_all.add_argument("--fact-top-k", type=int, default=6)
    p_all.add_argument("--top-k-per-item", type=int, default=4)

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    data_dir = Path(args.data_dir)
    persist_dir = Path(args.persist_dir)

    if args.command == "embed":
        generate_embeddings(
            data_dir=data_dir,
            persist_dir=persist_dir,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            rebuild=args.rebuild,
        )
        return

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run: python rag_cli.py embed --rebuild"
        )

    db = get_vectorstore(persist_dir, args.embedding_model)

    if args.command == "ask":
        run_qa(db, args.question, args.top_k)
    elif args.command == "fact":
        run_fact_check(db, args.text, args.top_k)
    elif args.command == "review":
        run_lit_review(db, args.outline, args.top_k_per_item)
    elif args.command == "all":
        run_all_in_one(
            db,
            question=args.question,
            claim_text=args.text,
            outline=args.outline,
            qa_top_k=args.top_k,
            fact_top_k=args.fact_top_k,
            review_top_k_per_item=args.top_k_per_item,
        )
    else:
        interactive_mode(db)


if __name__ == "__main__":
    main()
