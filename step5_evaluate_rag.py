import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from retrieval import retrieve_docs

CHAT_MODEL = "models/gemini-2.5-flash"


# -----------------------------
# Model + vectorstore helpers
# -----------------------------
def normalize_model_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"


def is_transient_error(error: Exception) -> bool:
    text = str(error).lower()
    transient_markers = [
        "503",
        "unavailable",
        "deadline_exceeded",
        "timeout",
        "temporarily",
        "connection reset",
        "rate limit",
        "resource exhausted",
    ]
    return any(marker in text for marker in transient_markers)


def retry_with_backoff(fn, max_retries: int = 5, base_delay: float = 1.5):
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if not is_transient_error(e) or attempt == max_retries - 1:
                raise
            sleep_seconds = base_delay * (2 ** attempt)
            print(
                f"⚠️ Transient API error: {e}. Retrying in {sleep_seconds:.1f}s "
                f"({attempt + 1}/{max_retries})..."
            )
            time.sleep(sleep_seconds)
    raise last_error


def ask_model(prompt: str, temperature: float = 0.0) -> Tuple[str, str]:
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=temperature)
    response = retry_with_backoff(lambda: llm.invoke(prompt))
    return str(response.content), CHAT_MODEL


def get_vectorstore(persist_dir: Path, embedding_model: str):
    embeddings = GoogleGenerativeAIEmbeddings(model=normalize_model_name(embedding_model))
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="literature",
    )


# -----------------------------
# Evaluation dataset
# -----------------------------
@dataclass
class EvalSample:
    sample_id: str
    question: str
    reference_answer: str
    gold_chunks: List[Dict[str, Any]]


def load_eval_dataset(path: Path) -> List[EvalSample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found at {path}. "
            "Create it as JSONL with one object per line."
        )

    samples: List[EvalSample] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {idx} in {path}: {e}") from e

            sample_id = str(item.get("id") or f"sample_{idx}")
            question = str(item.get("question", "")).strip()
            reference_answer = str(item.get("reference_answer", "")).strip()
            gold_chunks = item.get("gold_chunks", [])

            if not question:
                raise ValueError(f"Line {idx}: 'question' is required.")
            if not reference_answer:
                raise ValueError(f"Line {idx}: 'reference_answer' is required.")
            if not isinstance(gold_chunks, list) or not gold_chunks:
                raise ValueError(
                    f"Line {idx}: 'gold_chunks' must be a non-empty list. "
                    "Each item can include source/page/chunk_id."
                )

            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    question=question,
                    reference_answer=reference_answer,
                    gold_chunks=gold_chunks,
                )
            )

    if not samples:
        raise ValueError(f"No valid samples found in {path}.")

    return samples


# -----------------------------
# Retrieval metric helpers
# -----------------------------
def _norm_text(value: Any) -> str:
    return str(value).strip().lower()


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def metadata_page_to_human(metadata: Dict[str, Any]) -> Optional[int]:
    page_index = _to_int_or_none(metadata.get("page_index"))
    if page_index is not None:
        return page_index + 1

    page = _to_int_or_none(metadata.get("page"))
    if page is not None:
        return page + 1
    return None


def gold_match(gold: Dict[str, Any], retrieved_meta: Dict[str, Any]) -> bool:
    """
    Flexible matching:
    - if gold has source/page/chunk_id -> exact match on those provided fields
    - if gold has source/page -> match both
    - if gold has only source -> match source
    """
    g_source = _norm_text(gold.get("source", ""))
    g_page = _to_int_or_none(gold.get("page"))
    g_chunk = _to_int_or_none(gold.get("chunk_id"))

    r_source = _norm_text(retrieved_meta.get("source", ""))
    r_page = metadata_page_to_human(retrieved_meta)
    r_chunk = _to_int_or_none(retrieved_meta.get("chunk_id"))

    if g_source and g_source != r_source:
        return False
    if g_page is not None and g_page != r_page:
        return False
    if g_chunk is not None and g_chunk != r_chunk:
        return False
    return True


def is_relevant(retrieved_meta: Dict[str, Any], gold_chunks: List[Dict[str, Any]]) -> bool:
    return any(gold_match(g, retrieved_meta) for g in gold_chunks)


def precision_recall_mrr(retrieved_docs, gold_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    k = len(retrieved_docs)

    relevant_flags = [is_relevant(doc.metadata, gold_chunks) for doc in retrieved_docs]
    hits = sum(1 for flag in relevant_flags if flag)

    unique_gold_covered = set()
    for ridx, doc in enumerate(retrieved_docs):
        if not relevant_flags[ridx]:
            continue
        for gidx, g in enumerate(gold_chunks):
            if gold_match(g, doc.metadata):
                unique_gold_covered.add(gidx)

    precision = hits / k if k > 0 else 0.0
    recall = len(unique_gold_covered) / len(gold_chunks) if gold_chunks else 0.0

    rr = 0.0
    for rank, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            rr = 1.0 / rank
            break

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": rr,
        "hit_count": float(hits),
    }


# -----------------------------
# Generation + LLM-judge metrics
# -----------------------------
def format_context_for_generation(docs) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = metadata_page_to_human(doc.metadata)
        if page is None:
            page = "unknown"
        text = doc.page_content.strip().replace("\n", " ")
        blocks.append(f"[{i}] Source: {source}, page: {page}\nExcerpt: {text}")
    return "\n\n".join(blocks)


def generate_answer(question: str, context: str) -> Tuple[str, str]:
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
    return ask_model(prompt, temperature=0.0)


def _extract_first_json_block(text: str) -> Optional[str]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    return match.group(0)


def judge_generation(
    question: str,
    context: str,
    generated_answer: str,
    reference_answer: str,
) -> Tuple[Dict[str, float], str, str]:
    prompt = f"""
You are evaluating a RAG system output.
Score each metric from 0.0 to 1.0:
- faithfulness: How well answer claims are supported by the retrieved context only.
- answer_relevancy: How well the answer addresses the user question.
- answer_correctness: Semantic correctness against the reference answer.

Return STRICT JSON only with this schema:
{{
  "faithfulness": <float 0..1>,
  "answer_relevancy": <float 0..1>,
  "answer_correctness": <float 0..1>,
  "notes": "short reason"
}}

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{generated_answer}

Reference Answer:
{reference_answer}
"""

    raw, model_used = ask_model(prompt, temperature=0.0)

    block = _extract_first_json_block(raw)
    if block is None:
        raise ValueError(f"Judge model did not return JSON. Raw output: {raw}")

    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse judge JSON: {e}. Raw output: {raw}") from e

    def clamp_metric(name: str) -> float:
        value = float(parsed.get(name, 0.0))
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    scores = {
        "faithfulness": clamp_metric("faithfulness"),
        "answer_relevancy": clamp_metric("answer_relevancy"),
        "answer_correctness": clamp_metric("answer_correctness"),
    }
    return scores, raw, model_used


# -----------------------------
# Main evaluation flow
# -----------------------------
def evaluate(
    db,
    samples: List[EvalSample],
    top_k: int,
    candidate_multiplier: int,
    disable_mmr: bool,
) -> Dict[str, Any]:
    per_sample: List[Dict[str, Any]] = []

    for s in samples:
        retrieved_docs = retry_with_backoff(
            lambda: retrieve_docs(
                db,
                s.question,
                top_k=top_k,
                candidate_multiplier=candidate_multiplier,
                use_mmr=not disable_mmr,
            )
        )

        retrieval_scores = precision_recall_mrr(retrieved_docs, s.gold_chunks)
        context = format_context_for_generation(retrieved_docs)

        generated_answer, answer_model = generate_answer(s.question, context)

        judge_scores, judge_raw, judge_model = judge_generation(
            question=s.question,
            context=context,
            generated_answer=generated_answer,
            reference_answer=s.reference_answer,
        )

        row = {
            "id": s.sample_id,
            "question": s.question,
            "reference_answer": s.reference_answer,
            "generated_answer": generated_answer,
            "retrieval": retrieval_scores,
            "generation": judge_scores,
            "answer_model": answer_model,
            "judge_model": judge_model,
            "judge_raw": judge_raw,
            "retrieved": [
                {
                    "rank": rank,
                    "source": doc.metadata.get("source"),
                    "page": metadata_page_to_human(doc.metadata),
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
                for rank, doc in enumerate(retrieved_docs, start=1)
            ],
        }
        per_sample.append(row)

        print(f" Evaluated {s.sample_id} | P@{top_k}={retrieval_scores['precision_at_k']:.3f} "
              f"R@{top_k}={retrieval_scores['recall_at_k']:.3f} "
              f"MRR={retrieval_scores['mrr']:.3f} "
              f"Faith={judge_scores['faithfulness']:.3f} "
              f"Rel={judge_scores['answer_relevancy']:.3f} "
              f"Corr={judge_scores['answer_correctness']:.3f}")

    summary = {
        "num_samples": len(per_sample),
        "retrieval": {
            "precision_at_k": mean(r["retrieval"]["precision_at_k"] for r in per_sample),
            "recall_at_k": mean(r["retrieval"]["recall_at_k"] for r in per_sample),
            "mrr": mean(r["retrieval"]["mrr"] for r in per_sample),
        },
        "generation": {
            "faithfulness": mean(r["generation"]["faithfulness"] for r in per_sample),
            "answer_relevancy": mean(r["generation"]["answer_relevancy"] for r in per_sample),
            "answer_correctness": mean(r["generation"]["answer_correctness"] for r in per_sample),
        },
    }

    return {
        "summary": summary,
        "samples": per_sample,
    }


def maybe_write_template(path: Path):
    if path.exists():
        return

    template_rows = [
        {
            "id": "q1",
            "question": "What is the main contribution of the Transformer paper?",
            "reference_answer": "The Transformer replaces recurrence with self-attention and enables parallel sequence modeling.",
            "gold_chunks": [
                {"source": "1706.03762v7.pdf", "page": 1},
                {"source": "1706.03762v7.pdf", "page": 2},
            ],
        },
        {
            "id": "q2",
            "question": "What role does retrieval play in retrieval-augmented generation systems?",
            "reference_answer": "Retrieval provides external evidence so generation is grounded in relevant documents.",
            "gold_chunks": [
                {"source": "2312.10997v5.pdf", "page": 1},
            ],
        },
    ]

    with path.open("w", encoding="utf-8") as f:
        for row in template_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"📝 Created template evaluation set at: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 5 for RAG: Evaluate retrieval and generation quality."
    )
    parser.add_argument(
        "--eval-set",
        default="eval_dataset.jsonl",
        help="JSONL file with question/reference_answer/gold_chunks.",
    )
    parser.add_argument(
        "--persist-dir",
        default="vectorstore/chroma",
        help="Path to persisted Chroma DB.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--embedding-model",
        default="models/gemini-embedding-001",
        help="Embedding model used for vector retrieval.",
    )
    parser.add_argument(
        "--output",
        default="rag_eval_results.json",
        help="Where to save detailed evaluation results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional: limit number of samples for a quick run.",
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a starter eval_dataset.jsonl if it does not exist.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=5,
        help="Candidate expansion factor before reranking. Higher can improve recall at extra latency.",
    )
    parser.add_argument(
        "--disable-mmr",
        action="store_true",
        help="Disable MMR candidate union during retrieval reranking.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    eval_set_path = Path(args.eval_set)
    if args.create_template:
        maybe_write_template(eval_set_path)

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run step1_generate_embeddings.py first."
        )

    samples = load_eval_dataset(eval_set_path)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    db = get_vectorstore(persist_dir, args.embedding_model)

    results = evaluate(
        db=db,
        samples=samples,
        top_k=args.top_k,
        candidate_multiplier=max(2, args.candidate_multiplier),
        disable_mmr=args.disable_mmr,
    )

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = results["summary"]
    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved detailed results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
