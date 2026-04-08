import os
import re
import json
from pathlib import Path
from datetime import datetime

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


def run_qa(db, question: str, top_k: int, use_reranker: bool = False, use_mmr: bool = False):
    docs = retrieve_docs(db, question, top_k=top_k, use_reranker=use_reranker, use_mmr=use_mmr)
    if not docs:
        return "No relevant chunks found.", [], CHAT_MODEL

    context, sources = format_context(docs)
    prompt = build_qa_prompt(question, context)
    answer, used_model = ask_model(prompt, CHAT_MODEL)
    answer = enrich_citations_in_text(answer, sources)
    answer = append_citation_legend(answer, sources)
    return answer, sources, used_model


def run_fact_check(db, text: str, top_k: int, use_reranker: bool = False, use_mmr: bool = False):
    docs = retrieve_docs(db, text, top_k=top_k, use_reranker=use_reranker, use_mmr=use_mmr)
    if not docs:
        return "No relevant chunks found.", [], CHAT_MODEL

    context, sources = format_context(docs)
    prompt = build_fact_prompt(text, context)
    report, used_model = ask_model(prompt, CHAT_MODEL)
    report = enrich_citations_in_text(report, sources)
    report = append_citation_legend(report, sources)
    return report, sources, used_model


def run_lit_review(db, outline: str, top_k_per_item: int, use_reranker: bool = False, use_mmr: bool = False):
    outline_items = parse_outline(outline)
    collected = []
    seen = set()

    for item in outline_items:
        docs = retrieve_docs(db, item, top_k=top_k_per_item, use_reranker=use_reranker, use_mmr=use_mmr)
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


def get_chat_history_file():
    """Get the path to the chat history JSON file."""
    return Path("chat_history.json")


def load_chat_history():
    """Load chat history from file."""
    history_file = get_chat_history_file()
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                return json.load(f)
        except:
            return []
    return []


def save_chat_history(history):
    """Save chat history to file."""
    history_file = get_chat_history_file()
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def add_to_history(prompt: str, response: str, mode: str, sources: list):
    """Add a new entry to chat history."""
    history = load_chat_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response[:500],  # Store first 500 chars
        "mode": mode,
        "sources_count": len(sources)
    }
    history.append(entry)
    # Keep only last 50 conversations
    if len(history) > 50:
        history = history[-50:]
    save_chat_history(history)


def init_session_state():
    """Initialize session state variables."""
    if "persist_dir" not in st.session_state:
        st.session_state.persist_dir = "vectorstore/chroma"
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "models/gemini-embedding-001"
    if "qa_top_k" not in st.session_state:
        st.session_state.qa_top_k = 3
    if "fact_top_k" not in st.session_state:
        st.session_state.fact_top_k = 3
    if "review_top_k" not in st.session_state:
        st.session_state.review_top_k = 3
    if "use_reranker" not in st.session_state:
        st.session_state.use_reranker = False
    if "use_mmr" not in st.session_state:
        st.session_state.use_mmr = False


def main():
    st.set_page_config(
        page_title="📚 Literature Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 16px;
            padding: 10px 20px;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .chat-history-item {
            padding: 0.75rem;
            background-color: #f0f0f0;
            border-radius: 0.4rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 3px solid #1f77b4;
        }
        .chat-history-item:hover {
            background-color: #e8e8e8;
            transform: translateX(5px);
        }
        h1 {
            color: #1f77b4;
            text-align: center;
        }
        .model-badge {
            background-color: #e8f4f8;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            border-radius: 0.4rem;
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Header
    st.markdown("# 📚 Literature Assistant")
    st.markdown("### RAG-powered Q&A, Fact-checking & Literature Review")
    
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY is missing. Add it to .env file or set it in fly.io secrets.")
        st.stop()

    # Sidebar - Chat History & Settings
    with st.sidebar:
        st.markdown("---")
        
        # Settings Section (Collapsible)
        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.persist_dir = st.text_input(
                "Vector store path",
                value=st.session_state.persist_dir
            )
            st.session_state.embedding_model = st.text_input(
                "Embedding model",
                value=st.session_state.embedding_model
            )
            st.session_state.qa_top_k = st.slider("Q&A top-k", 1, 20, st.session_state.qa_top_k)
            st.session_state.fact_top_k = st.slider("Fact-check top-k", 1, 20, st.session_state.fact_top_k)
            st.session_state.review_top_k = st.slider("Review top-k", 1, 20, st.session_state.review_top_k)
            
            st.markdown("---")
            st.markdown("**Performance Options**")
            st.session_state.use_reranker = st.checkbox(
                "🚀 Enable LLM Reranker (slower but more accurate)",
                value=st.session_state.use_reranker,
                help="Uses Gemini to rerank results for higher precision"
            )
            st.session_state.use_mmr = st.checkbox(
                "🔀 Enable MMR Search (slower but more diverse)",
                value=st.session_state.use_mmr,
                help="Uses Max Marginal Relevance for diverse results"
            )
        
        st.markdown("---")
        
        # Chat History Section
        st.markdown("### 💬 Chat History")
        
        history = load_chat_history()
        
        if history:
            if st.button("🗑️ Clear History", use_container_width=True):
                save_chat_history([])
                st.rerun()
            
            st.markdown("---")
            
            for i, entry in reversed(list(enumerate(history[-10:]))):  # Show last 10
                timestamp = entry.get("timestamp", "")
                prompt_text = entry.get("prompt", "")[:40]
                mode = entry.get("mode", "ask")
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M")
                except:
                    time_str = "N/A"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        f"{mode.upper()} · {prompt_text}...",
                        key=f"history_{i}",
                        use_container_width=True
                    ):
                        st.session_state.selected_history = entry
                        st.rerun()
                with col2:
                    st.caption(time_str)
        else:
            st.info("No chat history yet. Start a conversation!")

    persist_dir = st.session_state.persist_dir
    embedding_model = st.session_state.embedding_model
    qa_top_k = st.session_state.qa_top_k
    fact_top_k = st.session_state.fact_top_k
    review_top_k = st.session_state.review_top_k
    use_reranker = st.session_state.use_reranker
    use_mmr = st.session_state.use_mmr

    if not Path(persist_dir).exists():
        st.error(f"❌ Vector store not found at: {persist_dir}")
        st.stop()

    db = load_db(persist_dir, embedding_model)

    # Main content
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Enter Your Query")
    with col2:
        mode_info = st.selectbox(
            "Mode",
            ["Q&A", "Fact-Check", "Literature Review"],
            label_visibility="collapsed"
        )

    user_prompt = st.text_area(
        "Your prompt",
        height=120,
        placeholder="💡 Ask a question, verify a claim, or request a literature review...",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submit_button = st.button("🚀 Submit", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.user_prompt = ""
        st.rerun()

    if submit_button and user_prompt.strip():
        with st.spinner("🔍 Processing your query..."):
            text = user_prompt.strip()
            q, claim_text, outline = parse_tagged_all_in_one(text)
            ran = False

            if q:
                answer, sources, used_model = run_qa(db, q, qa_top_k, use_reranker=use_reranker, use_mmr=use_mmr)
                add_to_history(q, answer, "qa", sources)
                
                st.markdown("---")
                st.markdown("### 📖 Answer")
                st.markdown(answer)
                
                with st.expander("📚 Sources"):
                    st.dataframe(source_table(sources), use_container_width=True)
                st.caption(f"🤖 Model: {used_model}")
                ran = True

            if claim_text:
                report, sources, used_model = run_fact_check(db, claim_text, fact_top_k, use_reranker=use_reranker, use_mmr=use_mmr)
                add_to_history(claim_text, report, "fact-check", sources)
                
                st.markdown("---")
                st.markdown("### ✅ Fact Check Report")
                st.markdown(report)
                
                with st.expander("📚 Sources"):
                    st.dataframe(source_table(sources), use_container_width=True)
                st.caption(f"🤖 Model: {used_model}")
                ran = True

            if outline:
                review, sources, used_model = run_lit_review(db, outline, review_top_k, use_reranker=use_reranker, use_mmr=use_mmr)
                add_to_history(outline, review, "literature-review", sources)
                
                st.markdown("---")
                st.markdown("### 📝 Literature Review")
                st.markdown(review)
                
                with st.expander("📚 Sources"):
                    st.dataframe(source_table(sources), use_container_width=True)
                st.caption(f"🤖 Model: {used_model}")
                ran = True

            if not ran:
                intent = detect_intent(text)
                
                if intent == "fact":
                    report, sources, used_model = run_fact_check(db, text, fact_top_k, use_reranker=use_reranker, use_mmr=use_mmr)
                    add_to_history(text, report, "fact-check", sources)
                    
                    st.markdown("---")
                    st.markdown("### ✅ Fact Check Report")
                    st.markdown(report)
                    
                elif intent == "review":
                    review, sources, used_model = run_lit_review(db, text, review_top_k, use_reranker=use_reranker, use_mmr=use_mmr)
                    add_to_history(text, review, "literature-review", sources)
                    
                    st.markdown("---")
                    st.markdown("### 📝 Literature Review")
                    st.markdown(review)
                    
                else:
                    answer, sources, used_model = run_qa(db, text, qa_top_k, use_reranker=use_reranker, use_mmr=use_mmr)
                    add_to_history(text, answer, "qa", sources)
                    
                    st.markdown("---")
                    st.markdown("### 📖 Answer")
                    st.markdown(answer)

                with st.expander("📚 Sources"):
                    st.dataframe(source_table(sources), use_container_width=True)
                st.caption(f"🤖 Model: {used_model}")
    
    elif submit_button:
        st.warning("⚠️ Please enter a prompt to continue.")


if __name__ == "__main__":
    main()
