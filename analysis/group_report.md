# Group Report — Lab 18: Production RAG

**Nhóm:** Bài làm cá nhân — Phan Văn Hiếu
**Ngày:** 18/08/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|---|---|:---:|---:|
| Phan Văn Hiếu | M1: Chunking | ✅ | 13/13 |
| Phan Văn Hiếu | M2: Hybrid Search | ✅ | 5/5 |
| Phan Văn Hiếu | M3: Reranking | ✅ | 5/5 |
| Phan Văn Hiếu | M4: Evaluation | ✅ | 4/4 |
| Phan Văn Hiếu | M5: Enrichment | ✅ | 10/10 |
| **Tổng** | **M1–M5** | **✅** | **37/37** |

## Kết quả chạy end-to-end

- Python: 3.11.15.
- Corpus: 26 tài liệu đọc được; 2 PDF scan bị bỏ qua đúng boundary vì không có text layer.
- Production chunking: 117 hierarchical child chunks.
- Qdrant: chạy local tại port 6333/6334.
- `python main.py`: exit code 0 sau khi sửa bước di chuyển report trên Windows từ `os.rename` sang `os.replace`.
- Hai report đã được tạo trong `reports/`.

## Kết quả RAGAS

| Metric | Naive | Production | Δ | Diễn giải |
|---|---:|---:|---:|---|
| Faithfulness | 0.0000 | 0.0000 | +0.0000 | Fallback, chưa đo |
| Answer Relevancy | 0.0000 | 0.0000 | +0.0000 | Fallback, chưa đo |
| Context Precision | 0.0000 | 0.0000 | +0.0000 | Fallback, chưa đo |
| Context Recall | 0.0000 | 0.0000 | +0.0000 | Fallback, chưa đo |

Không có `OPENAI_API_KEY`, nên M4 chủ động trả đủ bốn metric bằng 0 và `per_question=[]`. Đây là kiểm tra độ bền của pipeline, không phải benchmark so sánh chất lượng. BGE-M3 và bge-reranker-v2-m3 cũng chưa có trong cache; lần chạy nộp dùng hashing dense và lexical reranker fallback. Diagnostic pass thủ công được ghi trong `analysis/failure_analysis.md`.

## Key Findings

1. **Biggest improvement:** Pipeline đã có control point rõ ở cả năm module: chunk có parent/source, hybrid giữ metadata, rerank bảo toàn original score, evaluation không crash, enrichment combined chỉ thử một API call/chunk.
2. **Biggest challenge:** Môi trường không có model BGE/API key và kết nối tải model không ổn định. Fallback giúp chạy end-to-end nhưng không thay thế benchmark production thật.
3. **Surprise finding:** Giữ đúng nguồn vẫn chưa đủ để xử lý policy cũ/mới. Cần metadata có nghĩa (`version`, `effective_date`, `status`) và quy tắc filter/boost trước RRF.
4. **Failure quan trọng nhất:** Character-based child split có thể làm rơi từ phủ định “chưa”, khiến context đúng nguồn nhưng sai nghĩa. Đây là lỗi M1 có thể gây hậu quả lớn hơn việc retrieval thiếu một chunk.
5. **Câu hỏi đa-hop:** Các câu hỏi kết hợp phép + lương hoặc phần trăm + bảng lương cần coverage từ nhiều nguồn và một bước tổng hợp; top-1 context fallback không đủ.

## Presentation Notes

1. **RAGAS:** 0/0 vì fallback thiếu key; không tuyên bố production tốt hơn baseline khi chưa có metric thật.
2. **Biggest win:** Metadata được giữ xuyên M1 → M2 → M3 → M5, giúp diagnostic xác định chính xác source/version bị xếp sai.
3. **Case study:** “Bao nhiêu ngày phép năm?” lấy nghỉ không lương và policy v2023; fix bằng current-policy metadata filter và version-aware reranking.
4. **Debug thực tế:** Python 3.10 không đạt yêu cầu; pip/network timeout; Docker registry timeout; PowerShell cp1252; model không cache; `os.rename` không overwrite trên Windows.
5. **Next optimization:** Sentence-boundary child overlap, version filter, multi-query decomposition, sau đó tải model thật và chạy lại RAGAS.
