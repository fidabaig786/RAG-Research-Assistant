import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from retrieval import retrieve_docs

from rag_cli import (
    CHAT_MODEL,
    append_citation_legend,
    ask_model,
    enrich_citations_in_text,
    format_context,
    get_vectorstore,
    parse_outline,
)


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


def build_fact_prompt(text: str, context: str) -> str:
    return f"""
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


def build_review_prompt(outline: str, context: str) -> str:
    return f"""
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


def run_qa(db, question: str, top_k: int):
    docs = retrieve_docs(db, question, top_k=top_k)
    if not docs:
        return "No relevant chunks found.", [], CHAT_MODEL

    context, sources = format_context(docs)
    prompt = build_qa_prompt(question, context)
    answer, used_model = ask_model(prompt, CHAT_MODEL)
    answer = enrich_citations_in_text(answer, sources)
    answer = append_citation_legend(answer, sources)
    return answer, sources, used_model


def run_fact_check(db, text: str, top_k: int):
    docs = retrieve_docs(db, text, top_k=top_k)
    if not docs:
        return "No relevant chunks found.", [], CHAT_MODEL

    context, sources = format_context(docs)
    prompt = build_fact_prompt(text, context)
    report, used_model = ask_model(prompt, CHAT_MODEL)
    report = enrich_citations_in_text(report, sources)
    report = append_citation_legend(report, sources)
    return report, sources, used_model


def run_lit_review(db, outline: str, top_k_per_item: int):
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
        return "No relevant chunks found.", [], CHAT_MODEL

    context, sources = format_context(collected)
    prompt = build_review_prompt(outline, context)
    review, used_model = ask_model(prompt, CHAT_MODEL)
    review = enrich_citations_in_text(review, sources)
    review = append_citation_legend(review, sources)
    return review, sources, used_model


def parse_tagged_all_in_one(user_input: str):
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


def detect_intent(prompt: str) -> str:
    lowered = prompt.lower()

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
        return "fact"
    if any(k in lowered for k in review_keywords):
        return "review"
    return "ask"


def source_table(sources):
    return [{"ID": i, "Source": s, "Page": p} for i, s, p in sources]


@st.cache_resource
def load_db(persist_dir: str, embedding_model: str):
    return get_vectorstore(Path(persist_dir), embedding_model)


def main():
    st.set_page_config(page_title=" RAG Prototype", page_icon="📚", layout="wide")
    st.title("RAG Prototype")
    st.caption(f"Model: {CHAT_MODEL}")

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is missing. Add it to .env file.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        persist_dir = st.text_input("Vector store path", value="vectorstore/chroma")
        embedding_model = st.text_input("Embedding model", value="models/gemini-embedding-001")
        qa_top_k = st.number_input("Q&A top-k", min_value=1, max_value=20, value=3)
        fact_top_k = st.number_input("Fact-check top-k", min_value=1, max_value=20, value=3)
        review_top_k = st.number_input("Review top-k per item", min_value=1, max_value=20, value=3)

    if not Path(persist_dir).exists():
        st.error(f"Vector store not found at: {persist_dir}")
        st.stop()

    db = load_db(persist_dir, embedding_model)

    st.subheader("Single prompt (auto-routed)")
    st.write("Enter one prompt. The app auto-detects question/fact-check/review.")
    # st.write("Optional all-in-one format: `q: ... ; claim: ... ; review: ...`")

    user_prompt = st.text_area("Prompt", height=130, placeholder="Ask, verify a claim, or request a literature review...")

    if st.button("Run", type="primary"):
        text = user_prompt.strip()
        if not text:
            st.warning("Please enter a prompt.")
            st.stop()

        q, claim_text, outline = parse_tagged_all_in_one(text)
        ran = False

        if q:
            answer, sources, used_model = run_qa(db, q, qa_top_k)
            st.markdown("### Answer")
            st.write(answer)
            st.caption(f"Model used: {used_model}")
            st.dataframe(source_table(sources), use_container_width=True)
            ran = True

        if claim_text:
            report, sources, used_model = run_fact_check(db, claim_text, fact_top_k)
            st.markdown("### Fact Check Report")
            st.write(report)
            st.caption(f"Model used: {used_model}")
            st.dataframe(source_table(sources), use_container_width=True)
            ran = True

        if outline:
            review, sources, used_model = run_lit_review(db, outline, review_top_k)
            st.markdown("### Literature Review Draft")
            st.write(review)
            st.caption(f"Model used: {used_model}")
            st.dataframe(source_table(sources), use_container_width=True)
            ran = True

        if not ran:
            intent = detect_intent(text)
            if intent == "fact":
                report, sources, used_model = run_fact_check(db, text, fact_top_k)
                st.markdown("### Fact Check Report")
                st.write(report)
            elif intent == "review":
                review, sources, used_model = run_lit_review(db, text, review_top_k)
                st.markdown("### Literature Review Draft")
                st.write(review)
            else:
                answer, sources, used_model = run_qa(db, text, qa_top_k)
                st.markdown("### Answer")
                st.write(answer)

            st.caption(f"Model used: {used_model}")
            st.dataframe(source_table(sources), use_container_width=True)


if __name__ == "__main__":
    main()
