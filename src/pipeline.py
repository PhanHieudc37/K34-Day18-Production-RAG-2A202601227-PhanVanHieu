from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from src.llm_provider import create_chat_client, get_llm_settings, has_llm_credentials
from config import RERANK_TOP_K


ANSWER_SYSTEM_PROMPT = """Trả lời CHỈ dựa trên context được cung cấp.
Trả lời đầy đủ mọi ý trong câu hỏi và thực hiện phép tính đơn giản khi context có đủ dữ kiện.
Khi các phiên bản mâu thuẫn, ưu tiên văn bản ghi rõ là hiện hành hoặc thay thế phiên bản cũ.
Phân biệt đúng loại quy trình (ví dụ mua sắm khác tạm ứng).
Nếu context thực sự không có bằng chứng thì trả lời đúng: Không tìm thấy."""


def generate_answer(query: str, contexts: list[str]) -> str:
    """Generate one grounded answer with a deterministic provider fallback."""
    if not contexts:
        return "Không tìm thấy."
    if not has_llm_credentials():
        return contexts[0]

    client, model = create_chat_client()
    context_str = "\n\n".join(contexts)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
        ],
        temperature=0,
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip() or "Không tìm thấy."


def build_pipeline():
    """Build production RAG pipeline."""
    timings = {}
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_text_by_id = {parent.parent_id: parent.text for parent in parents}
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id,
                    # Retrieve/rerank the precise child, but return its complete
                    # parent as context. This prevents tables and policy clauses
                    # from being cut at the 256-character child boundary.
                    "parent_text": parent_text_by_id[child.parent_id],
                },
            })
    timings["chunking"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({timings['chunking']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        timings["enrichment"] = time.time() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({timings['enrichment']:.1f}s)", flush=True)
    else:
        timings["enrichment"] = time.time() - t0
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    timings["indexing"] = time.time() - t0
    print(f"  ✓ Indexed ({timings['indexing']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    timings["reranker_load"] = time.time() - t0
    print(f"  ✓ Reranker ready ({timings['reranker_load']:.1f}s)", flush=True)

    search.lab18_timings = timings

    return search, reranker


def run_query(
    query: str,
    search: HybridSearch,
    reranker: CrossEncoderReranker,
    timings: dict[str, float] | None = None,
) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    timings = timings if timings is not None else {}
    t0 = time.time()
    results = search.search(query)
    timings["retrieval"] = timings.get("retrieval", 0.0) + time.time() - t0
    # Multiple child chunks from one parent otherwise consume all three rerank
    # slots. Keep the best RRF-ranked child per parent so multi-document
    # questions can still retrieve evidence from more than one policy.
    diverse_results = []
    seen_parents = set()
    for result in results:
        parent_key = result.metadata.get("parent_id") or (
            result.metadata.get("source"), result.text
        )
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        diverse_results.append(result)

    docs = [
        {"text": r.text, "score": r.score, "metadata": r.metadata}
        for r in diverse_results
    ]
    t0 = time.time()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    timings["reranking"] = timings.get("reranking", 0.0) + time.time() - t0
    selected = reranked if reranked else diverse_results[:RERANK_TOP_K]
    contexts = []
    seen_contexts = set()
    for result in selected:
        context = str(result.metadata.get("parent_text") or result.text)
        if context not in seen_contexts:
            contexts.append(context)
            seen_contexts.add(context)

    t0 = time.time()
    if has_llm_credentials() and contexts:
        try:
            answer = generate_answer(query, contexts)
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["generation"] = timings.get("generation", 0.0) + time.time() - t0
    return answer, contexts


def save_latency_report(
    timings: dict[str, float],
    num_queries: int,
    path: str = "analysis/latency_breakdown.md",
) -> None:
    """Persist measured stage latency for the rubric's reproducibility bonus."""
    settings = get_llm_settings()
    provider = settings.provider if settings else "offline fallback"
    per_query_steps = {"retrieval", "reranking", "generation"}
    labels = {
        "chunking": "M1 chunking",
        "enrichment": "M5 enrichment",
        "indexing": "M2 BM25 + dense indexing",
        "reranker_load": "M3 reranker load",
        "retrieval": "M2 hybrid retrieval",
        "reranking": "M3 reranking",
        "generation": "LLM answer generation",
        "ragas": "M4 RAGAS evaluation",
    }
    ordered_steps = list(labels)
    rows = []
    for step in ordered_steps:
        total = timings.get(step, 0.0)
        calls = num_queries if step in per_query_steps else 1
        average_ms = total * 1000 / max(calls, 1)
        rows.append(f"| {labels[step]} | {calls} | {total:.3f} | {average_ms:.2f} |")

    measured_total = sum(timings.get(step, 0.0) for step in ordered_steps)
    content = "\n".join([
        "# Latency Breakdown — Production RAG",
        "",
        f"- Provider: **{provider}**",
        f"- Evaluation queries: **{num_queries}**",
        "- Measurement: `time.time()` wall-clock, measured by the submission pipeline.",
        "- Model downloads are excluded; reranker/model loading is reported separately.",
        "",
        "| Stage | Calls | Total (s) | Average (ms/call) |",
        "|---|---:|---:|---:|",
        *rows,
        f"| **Measured total** | — | **{measured_total:.3f}** | — |",
        "",
    ])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Latency report saved to {path}")


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    timings = dict(getattr(search, "lab18_timings", {}))
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    os.makedirs("reports", exist_ok=True)
    with open("reports/production_eval_dataset.json", "w", encoding="utf-8") as file:
        json.dump({
            "questions": questions,
            "answers": answers,
            "contexts": all_contexts,
            "ground_truths": ground_truths,
        }, file, ensure_ascii=False, indent=2)
    print("  Evaluation dataset cached at reports/production_eval_dataset.json", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    timings["ragas"] = time.time() - t0
    print(f"  ✓ RAGAS done ({timings['ragas']:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    save_latency_report(timings, len(test_set))
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
