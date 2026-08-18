from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re, hashlib
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_UNAVAILABLE = False


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip()
    ]
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n+", text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    # A one-sentence document is already a complete semantic unit and does not
    # need an embedding call.
    if len(sentences) == 1:
        return [Chunk(
            text=sentences[0],
            metadata={**metadata, "strategy": "semantic", "chunk_index": 0},
        )]

    try:
        from sentence_transformers import SentenceTransformer
        from numpy import dot
        from numpy.linalg import norm

        global _SEMANTIC_MODEL, _SEMANTIC_MODEL_UNAVAILABLE
        if _SEMANTIC_MODEL_UNAVAILABLE:
            raise RuntimeError("semantic model was unavailable earlier in this process")
        if _SEMANTIC_MODEL is None:
            _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = _SEMANTIC_MODEL.encode(sentences, show_progress_bar=False)

        groups = [[sentences[0]]]
        for index in range(1, len(sentences)):
            left, right = embeddings[index - 1], embeddings[index]
            similarity = float(dot(left, right) / (norm(left) * norm(right) + 1e-9))
            if similarity < threshold:
                groups.append([sentences[index]])
            else:
                groups[-1].append(sentences[index])
    except Exception as exc:
        # Keep the pipeline usable when the lab machine cannot download models.
        # Paragraph/sentence units preserve author-defined ideas safely.
        _SEMANTIC_MODEL_UNAVAILABLE = True
        print(f"  Semantic model unavailable; using paragraph fallback: {exc}")
        groups = [[paragraph] for paragraph in paragraphs]

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
        if group
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    def split_oversized(value: str, max_size: int) -> list[str]:
        """Split a large unit near whitespace, with a hard-size fallback."""
        parts = []
        remaining = value.strip()
        while len(remaining) > max_size:
            cut = max(
                remaining.rfind("\n", 0, max_size + 1),
                remaining.rfind(" ", 0, max_size + 1),
            )
            if cut <= 0:
                cut = max_size
            parts.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            parts.append(remaining)
        return parts

    def group_to_size(value: str, max_size: int) -> list[str]:
        raw_units = [
            unit.strip()
            for unit in re.split(r"\n\s*\n+", value)
            if unit.strip()
        ]
        units = [part for unit in raw_units for part in split_oversized(unit, max_size)]
        grouped, current = [], ""
        for unit in units:
            candidate = f"{current}\n\n{unit}" if current else unit
            if current and len(candidate) > max_size:
                grouped.append(current)
                current = unit
            else:
                current = candidate
        if current:
            grouped.append(current)
        return grouped

    parent_texts = group_to_size(text, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", ""))

    for parent_index, parent_text in enumerate(parent_texts):
        digest = hashlib.sha1(
            f"{source}\0{parent_index}\0{parent_text}".encode("utf-8")
        ).hexdigest()[:16]
        parent_id = f"parent_{digest}"
        parent_metadata = {
            **metadata,
            "strategy": "hierarchical",
            "chunk_type": "parent",
            "chunk_index": parent_index,
            "parent_id": parent_id,
        }
        parents.append(Chunk(
            text=parent_text,
            metadata=parent_metadata,
            parent_id=parent_id,
        ))

        for child_index, child_text in enumerate(group_to_size(parent_text, child_size)):
            child_metadata = {
                **metadata,
                "strategy": "hierarchical",
                "chunk_type": "child",
                "chunk_index": len(children),
                "child_index": child_index,
                "parent_id": parent_id,
            }
            children.append(Chunk(
                text=child_text,
                metadata=child_metadata,
                parent_id=parent_id,
            ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text.strip():
        return []

    sections: list[tuple[str, str]] = []
    heading_path: list[str] = []
    current_lines: list[str] = []
    current_section = str(metadata.get("section", "preamble"))
    in_fence = False

    def flush_section() -> None:
        nonlocal current_lines
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_section, section_text))
        current_lines = []

    for line in text.splitlines():
        stripped = line.lstrip()
        fence_line = bool(re.match(r"^(```|~~~)", stripped))
        heading_match = None if in_fence else re.match(r"^(#{1,3})\s+(.+?)\s*$", line)

        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_path = heading_path[:level - 1]
            heading_path.append(title)
            current_section = " > ".join(heading_path)
            # Keep the heading in retrievable text as well as metadata.
            current_lines = [line]
        else:
            current_lines.append(line)

        if fence_line:
            in_fence = not in_fence

    flush_section()
    return [
        Chunk(
            text=section_text,
            metadata={
                **metadata,
                "strategy": "structure",
                "section": section,
                "chunk_index": index,
            },
        )
        for index, (section, section_text) in enumerate(sections)
    ]


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
