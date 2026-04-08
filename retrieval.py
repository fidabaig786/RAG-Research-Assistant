import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
    "why",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)?", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _doc_key(doc: Document) -> Tuple[str, Any, Any, int]:
    metadata = doc.metadata or {}
    return (
        str(metadata.get("source", "")),
        metadata.get("page_index", metadata.get("page")),
        metadata.get("chunk_id"),
        hash(doc.page_content),
    )


def _semantic_candidates(db, query: str, k: int) -> List[Tuple[Document, float]]:
    try:
        scored = db.similarity_search_with_relevance_scores(query, k=k)
        return [(doc, float(score)) for doc, score in scored]
    except Exception:
        pass

    try:
        scored = db.similarity_search_with_score(query, k=k)
        # Chroma often returns distance where lower is better.
        return [(doc, 1.0 / (1.0 + float(score))) for doc, score in scored]
    except Exception:
        docs = db.similarity_search(query, k=k)
        return [(doc, 0.5) for doc in docs]


def _lexical_score(query_terms: Iterable[str], text: str, source: str) -> float:
    terms = set(query_terms)
    if not terms:
        return 0.0

    body_tokens = set(_tokenize(text))
    source_tokens = set(_tokenize(source.replace(".pdf", " ")))

    overlap_body = len(terms & body_tokens) / len(terms)
    overlap_source = len(terms & source_tokens) / len(terms)

    query_numbers = {t for t in terms if t.isdigit()}
    body_numbers = {t for t in body_tokens if t.isdigit()}
    numeric_boost = 0.15 if query_numbers and (query_numbers & body_numbers) else 0.0

    score = 0.85 * overlap_body + 0.15 * overlap_source + numeric_boost
    return min(1.0, max(0.0, score))


def retrieve_docs(
    db,
    query: str,
    top_k: int,
    candidate_multiplier: int = 2,
    use_mmr: bool = False,
    min_combined_score: float = 0.38,
    relative_score_margin: float = 0.22,
    min_results: int = 2,
    use_reranker: bool = False,
    reranker_strategy: str = "llm",
) -> List[Document]:
    """
    Hybrid retrieval tuned for higher precision while keeping recall stable:
    1) Retrieve a wider semantic candidate pool.
    2) Optionally union with MMR candidates for coverage.
    3) Re-rank by combined semantic + lexical score.
    4) Apply light source/page diversity during top-k selection.
    5) Optionally use a dedicated reranker for final precision boost.
    """
    if top_k <= 0:
        return []

    candidate_k = max(top_k * max(2, candidate_multiplier), top_k)
    query_terms = _tokenize(query)

    candidate_map: Dict[Tuple[str, Any, Any, int], Dict[str, Any]] = {}

    for doc, sem_score in _semantic_candidates(db, query, candidate_k):
        key = _doc_key(doc)
        row = candidate_map.get(key)
        if row is None:
            row = {"doc": doc, "semantic": sem_score, "from_mmr": False}
            candidate_map[key] = row
        else:
            row["semantic"] = max(row["semantic"], sem_score)

    if use_mmr:
        try:
            mmr_k = min(candidate_k, max(top_k * 3, top_k + 2))
            mmr_docs = db.max_marginal_relevance_search(query, k=mmr_k, fetch_k=candidate_k)
            for doc in mmr_docs:
                key = _doc_key(doc)
                row = candidate_map.get(key)
                if row is None:
                    candidate_map[key] = {"doc": doc, "semantic": 0.40, "from_mmr": True}
                else:
                    row["from_mmr"] = True
        except Exception:
            pass

    if not candidate_map:
        return []

    scored_rows = []
    for row in candidate_map.values():
        doc = row["doc"]
        metadata = doc.metadata or {}
        source = str(metadata.get("source", ""))
        lexical = _lexical_score(query_terms, doc.page_content, source)

        semantic = max(0.0, min(1.0, float(row["semantic"])))
        combined = 0.75 * semantic + 0.25 * lexical
        scored_rows.append(
            {
                "doc": doc,
                "combined": combined,
                "source": source,
                "page": metadata.get("page_index", metadata.get("page")),
            }
        )

    scored_rows.sort(key=lambda x: x["combined"], reverse=True)

    # Precision-first filtering: keep only candidates near the best score.
    top_score = scored_rows[0]["combined"]
    relative_cutoff = max(0.0, top_score * (1.0 - relative_score_margin))
    hard_cutoff = max(min_combined_score, relative_cutoff)
    filtered_rows = [row for row in scored_rows if row["combined"] >= hard_cutoff]
    if len(filtered_rows) >= min_results:
        scored_rows = filtered_rows

    # Light diversity to avoid one source/page dominating final context.
    selected: List[Document] = []
    source_counts = defaultdict(int)
    source_page_counts = defaultdict(int)

    strong_rows = [row for row in scored_rows if row["combined"] >= (top_score - 0.10)]
    if len(strong_rows) >= min_results:
        target_n = min(top_k, len(strong_rows))
    else:
        target_n = min(top_k, len(scored_rows))

    while scored_rows and len(selected) < target_n:
        best_idx = 0
        best_score = float("-inf")

        for idx, row in enumerate(scored_rows):
            source = row["source"]
            source_page = (row["source"], row["page"])
            adjusted = row["combined"] - 0.06 * source_counts[source] - 0.08 * source_page_counts[source_page]
            if adjusted > best_score:
                best_idx = idx
                best_score = adjusted

        chosen = scored_rows.pop(best_idx)
        source = chosen["source"]
        source_page = (chosen["source"], chosen["page"])
        source_counts[source] += 1
        source_page_counts[source_page] += 1
        selected.append(chosen["doc"])

    # Optional: Apply reranker for final precision boost
    if use_reranker and selected:
        try:
            from reranker import LLMReranker
            reranker = LLMReranker()
            selected = reranker.rerank(query, selected, top_k=top_k)
        except Exception as e:
            print(f"Warning: Reranker failed ({e}), using base results")
    
    return selected
