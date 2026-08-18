from __future__ import annotations

"""Module 5: controlled chunk enrichment with offline fallbacks."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def _extractive_summary(text: str) -> str:
    sentences = _sentences(text)
    return " ".join(sentences[:2]).strip() if sentences else text.strip()


def _extractive_questions(text: str, n_questions: int) -> list[str]:
    if n_questions <= 0:
        return []
    candidates = [sentence for sentence in _sentences(text) if len(sentence) > 10]
    return [f"{sentence.rstrip('.!?')}?" for sentence in candidates[:n_questions]]


def _context_line(document_title: str) -> str:
    if document_title:
        return f"Trích từ tài liệu {document_title}."
    return "Đoạn trích từ tài liệu chính sách."


def _local_metadata(text: str) -> dict:
    lowered = text.lower()
    if any(word in lowered for word in ("mật khẩu", "vpn", "mfa", "malware", "bảo mật")):
        category = "it"
    elif any(word in lowered for word in ("chi phí", "tạm ứng", "mua sắm", "lương", "triệu")):
        category = "finance"
    elif any(word in lowered for word in ("nhân viên", "nghỉ", "thử việc", "bảo hiểm")):
        category = "hr"
    else:
        category = "policy"

    heading = re.search(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE)
    if heading:
        topic = heading.group(1).strip()
    else:
        words = re.sub(r"\s+", " ", text).strip().split()
        topic = " ".join(words[:12]) if words else "general"

    entities = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", text)))
    has_vietnamese = bool(re.search(r"[À-ỹĐđ]", text))
    return {
        "topic": topic,
        "entities": entities,
        "category": category,
        "language": "vi" if has_vietnamese else "en",
    }


def _parse_json_object(content: str) -> dict:
    match = re.search(r"\{.*\}", content.strip(), flags=re.DOTALL)
    if not match:
        raise ValueError("model response does not contain a JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def summarize_chunk(text: str) -> str:
    """Return a concise Vietnamese summary, or an extractive local fallback."""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            summary = (response.choices[0].message.content or "").strip()
            if summary:
                return summary
        except Exception as exc:
            print(f"  OpenAI summarize failed; using local fallback: {exc}")
    return _extractive_summary(text)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that the chunk can answer."""
    if n_questions <= 0:
        return []
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                            "Mỗi câu hỏi trên một dòng."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            content = response.choices[0].message.content or ""
            questions = []
            for line in content.splitlines():
                question = re.sub(r"^\s*(?:[-*]|\d+[.)-]?)\s*", "", line).strip()
                if question:
                    questions.append(question)
            if questions:
                return questions[:n_questions]
        except Exception as exc:
            print(f"  OpenAI HyQA failed; using local fallback: {exc}")
    return _extractive_questions(text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a one-line document context while preserving the original text."""
    context = ""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu "
                            "và nói về chủ đề gì. Chỉ trả về một câu."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=80,
            )
            context = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  OpenAI contextual prepend failed; using local fallback: {exc}")
    if not context:
        context = _context_line(document_title)
    return f"{context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract topic, entities, category and language metadata."""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            'Trả về JSON: {"topic":"...","entities":[],"category":'
                            '"policy|hr|it|finance","language":"vi|en"}.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
            )
            metadata = _parse_json_object(response.choices[0].message.content or "")
            if metadata:
                return metadata
        except Exception as exc:
            print(f"  OpenAI metadata extraction failed; using local fallback: {exc}")
    return _local_metadata(text)


def _fallback_combined(text: str, source: str) -> dict:
    return {
        "summary": _extractive_summary(text),
        "questions": _extractive_questions(text, 3),
        "context": _context_line(source),
        "metadata": _local_metadata(text),
    }


def _normalize_combined(result: dict, text: str, source: str) -> dict:
    fallback = _fallback_combined(text, source)
    summary = result.get("summary")
    context = result.get("context")
    questions = result.get("questions")
    metadata = result.get("metadata")
    return {
        "summary": summary.strip() if isinstance(summary, str) and summary.strip() else fallback["summary"],
        "questions": (
            [str(question).strip() for question in questions if str(question).strip()][:3]
            if isinstance(questions, list) and questions
            else fallback["questions"]
        ),
        "context": context.strip() if isinstance(context, str) and context.strip() else fallback["context"],
        "metadata": metadata if isinstance(metadata, dict) else fallback["metadata"],
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Return summary, questions, context and metadata using at most one API call."""
    if not OPENAI_API_KEY:
        return _fallback_combined(text, source)

    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Phân tích đoạn văn và trả về một JSON object gồm: "
                        '"summary" (2-3 câu), "questions" (3 câu hỏi), '
                        '"context" (một câu mô tả vị trí/chủ đề), và "metadata" '
                        'với topic, entities, category, language.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        parsed = _parse_json_object(response.choices[0].message.content or "")
        return _normalize_combined(parsed, text, source)
    except Exception as exc:
        # Do not call the four individual LLM functions here: combined mode is
        # deliberately capped at one attempted API call per chunk.
        print(f"  Combined enrichment failed; using local fallback: {exc}")
        return _fallback_combined(text, source)


def _compose_enriched_text(
    original_text: str,
    summary: str,
    questions: list[str],
    context: str,
) -> str:
    parts = []
    if context.strip():
        parts.append(context.strip())
    if summary.strip():
        parts.append(f"Tóm tắt: {summary.strip()}")
    if questions:
        question_lines = "\n".join(f"- {question}" for question in questions)
        parts.append(f"Câu hỏi có thể trả lời:\n{question_lines}")
    parts.append(original_text)
    return "\n\n".join(parts)


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Enrich chunks while preserving their original text and source metadata."""
    selected_methods = list(methods) if methods is not None else ["combined"]
    use_combined = "combined" in selected_methods
    enriched = []

    for index, chunk in enumerate(chunks):
        text = chunk["text"]
        original_metadata = dict(chunk.get("metadata", {}))
        source = str(original_metadata.get("source", ""))

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result["summary"]
            questions = result["questions"]
            context = result["context"]
            auto_metadata = result["metadata"]
            enriched_text = _compose_enriched_text(text, summary, questions, context)
        else:
            summary = summarize_chunk(text) if "summary" in selected_methods else ""
            questions = (
                generate_hypothesis_questions(text)
                if "hyqa" in selected_methods
                else []
            )
            contextual_text = (
                contextual_prepend(text, source)
                if "contextual" in selected_methods
                else text
            )
            context = ""
            if contextual_text.endswith(text):
                context = contextual_text[:-len(text)].strip()
            auto_metadata = (
                extract_metadata(text)
                if "metadata" in selected_methods
                else {}
            )
            enriched_text = _compose_enriched_text(text, summary, questions, context)

        # Original metadata wins on conflicts, especially source and version.
        merged_metadata = {**dict(auto_metadata), **original_metadata}
        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=list(questions),
            auto_metadata=merged_metadata,
            method="+".join(selected_methods) if selected_methods else "none",
        ))

        if (index + 1) % 10 == 0 or (index + 1) == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ tăng thêm theo thâm niên công tác."
    )
    result = enrich_chunks([
        {"text": sample, "metadata": {"source": "nghi_phep_nam_v2024.md"}},
    ])[0]
    print(result)
