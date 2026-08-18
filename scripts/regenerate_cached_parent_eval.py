"""Regenerate cached answers after resolving hierarchical child hits to parents.

This utility avoids repeating embedding/index/reranking when only the
child-to-parent context handoff or answer prompt changed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.m1_chunking import chunk_hierarchical, load_documents
from src.pipeline import generate_answer


CACHE_PATH = Path("reports/production_eval_dataset.json")


def main() -> None:
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    parent_text_by_id: dict[str, str] = {}
    children = []
    for document in load_documents():
        parents, document_children = chunk_hierarchical(
            document["text"], metadata=document["metadata"]
        )
        parent_text_by_id.update({parent.parent_id: parent.text for parent in parents})
        children.extend(document_children)

    expanded_contexts: list[list[str]] = []
    for cached_contexts in data["contexts"]:
        resolved = []
        for cached_context in cached_contexts:
            matches = [child for child in children if child.text in cached_context]
            if not matches:
                raise RuntimeError("Cannot resolve a cached child context to its parent")
            child = max(matches, key=lambda item: len(item.text))
            parent_text = parent_text_by_id[child.parent_id]
            if parent_text not in resolved:
                resolved.append(parent_text)
        expanded_contexts.append(resolved)

    answers = []
    last_started = 0.0
    for index, (question, contexts) in enumerate(
        zip(data["questions"], expanded_contexts), start=1
    ):
        # 12 RPM leaves headroom below Gemini's common free-tier 15 RPM limit.
        remaining = 5.0 - (time.monotonic() - last_started)
        if remaining > 0:
            time.sleep(remaining)
        last_started = time.monotonic()
        answers.append(generate_answer(question, contexts))
        print(f"Generated {index}/{len(data['questions'])}", flush=True)

    data["contexts"] = expanded_contexts
    data["answers"] = answers
    CACHE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Updated {CACHE_PATH}")


if __name__ == "__main__":
    main()
