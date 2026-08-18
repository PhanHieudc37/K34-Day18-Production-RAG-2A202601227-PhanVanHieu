from __future__ import annotations

"""Module 4: RAGAS evaluation — four metrics and failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAGAS_MAX_WORKERS, TEST_SET_PATH
from src.llm_provider import (
    create_ragas_models,
    get_llm_settings,
    has_llm_credentials,
)


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load the evaluation questions and ground truths from JSON."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _safe_float(value) -> float:
    """Convert metric output to a finite JSON-safe float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _empty_evaluation() -> dict:
    return {**{metric: 0.0 for metric in METRIC_NAMES}, "per_question": []}


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS while keeping API/dependency failures non-fatal."""
    try:
        lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
        if len(lengths) != 1:
            raise ValueError("questions, answers, contexts and ground_truths must be parallel")
        if not questions:
            return _empty_evaluation()
        if not has_llm_credentials():
            raise RuntimeError("No OpenAI or Gemini API key is configured")

        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        ragas_llm, ragas_embeddings = create_ragas_models()
        settings = get_llm_settings()
        if settings and settings.provider == "gemini":
            # Gemini's OpenAI-compatible endpoint does not support n > 1.
            # RAGAS defaults to strictness=3 (three generated candidates), so
            # use one candidate instead of silently producing a zero metric.
            answer_relevancy.strictness = 1
        default_workers = 1 if settings and settings.provider == "gemini" else 4
        max_workers = RAGAS_MAX_WORKERS if RAGAS_MAX_WORKERS > 0 else default_workers
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(
                timeout=180,
                max_retries=5,
                max_wait=30,
                max_workers=max_workers,
            ),
        )
        dataframe = result.to_pandas()
        per_question = []
        for _, row in dataframe.iterrows():
            row_contexts = row.get("contexts", [])
            if not isinstance(row_contexts, list):
                try:
                    row_contexts = list(row_contexts)
                except TypeError:
                    row_contexts = [str(row_contexts)] if row_contexts else []
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=[str(context) for context in row_contexts],
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            ))

        aggregates = {}
        for metric in METRIC_NAMES:
            values = [getattr(item, metric) for item in per_question]
            aggregates[metric] = sum(values) / len(values) if values else 0.0
        return {**aggregates, "per_question": per_question}
    except Exception as exc:
        print(f"  RAGAS evaluation unavailable; using zero-score fallback: {exc}")
        return _empty_evaluation()


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Return the lowest-average questions with a metric-specific diagnosis."""
    if bottom_n <= 0 or not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "The answer contains claims unsupported by the retrieved context.",
            "Tighten the context-only prompt and lower generation temperature.",
        ),
        "answer_relevancy": (
            "The answer does not directly address the question.",
            "Improve the answer prompt and require a concise, question-focused response.",
        ),
        "context_precision": (
            "The retrieved context contains too many irrelevant chunks.",
            "Improve reranking or add source/version metadata filters.",
        ),
        "context_recall": (
            "The retrieved context is missing information needed for the answer.",
            "Improve chunking and retrieve more candidates with BM25 plus dense search.",
        ),
    }

    analyzed = []
    for item in eval_results:
        metric_scores = {
            metric: _safe_float(getattr(item, metric))
            for metric in METRIC_NAMES
        }
        worst_metric = min(metric_scores, key=metric_scores.get)
        average_score = sum(metric_scores.values()) / len(metric_scores)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": item.question,
            "answer": item.answer,
            "contexts": list(item.contexts),
            "ground_truth": item.ground_truth,
            "score": average_score,
            "average_score": average_score,
            "worst_metric": worst_metric,
            "worst_metric_score": metric_scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda failure: failure["average_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save aggregate metrics and diagnosed failures to JSON."""
    report = {
        "aggregate": {key: value for key, value in results.items() if key != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
