# Group Report — Lab 18: Production RAG

**Hình thức:** Bài làm cá nhân

**Sinh viên:** Phan Văn Hiếu

**Ngày:** 18/08/2026

## Phân công và kiểm thử

| Người thực hiện | Module | Trạng thái | Tests |
|---|---|:---:|---:|
| Phan Văn Hiếu | M1 Chunking | ✅ | 13/13 |
| Phan Văn Hiếu | M2 Hybrid Search | ✅ | 5/5 |
| Phan Văn Hiếu | M3 Reranking | ✅ | 5/5 |
| Phan Văn Hiếu | M4 RAGAS + Failure Analysis | ✅ | 4/4 |
| Phan Văn Hiếu | M5 Enrichment | ✅ | 10/10 |
| **Tổng** | **M1–M5** | **✅** | **37/37** |

## End-to-end

- Python 3.11.15; Qdrant local tại port 6333.
- 26 tài liệu có text; 2 PDF scan được cảnh báo và bỏ qua đúng boundary chưa OCR.
- M1 tạo 117 hierarchical child chunks, mỗi child liên kết stable `parent_id`.
- M5 combined mode: một API call/chunk, fallback local và original metadata thắng khi merge.
- M2 BM25 tiếng Việt + BGE-M3/Qdrant + RRF; payload giữ text/source/parent.
- M3 dùng `BAAI/bge-reranker-v2-m3`, top-3 giảm dần, giữ original score/metadata/rank.
- Runtime cuối retrieve/rerank child nhưng trả parent đầy đủ và deduplicate parent để tăng source coverage.
- `python main.py` đã exit 0 và sinh đủ hai report; model cache được xác minh offline.

## RAGAS thật trên 20 câu

| Metric | Naive | Production | Δ | Rubric |
|---|---:|---:|---:|---|
| Faithfulness | 0.3875 | **0.9250** | +0.5375 | bonus ≥ 0.85 |
| Answer Relevancy | 0.5435 | **0.8835** | +0.3401 | đạt ≥ 0.75 |
| Context Precision | 0.5250 | **0.7750** | +0.2500 | đạt ≥ 0.75 |
| Context Recall | 0.5750 | **0.9250** | +0.3500 | đạt ≥ 0.75 |

Production đạt 4/4 metric ≥ 0.75. Report được tạo bằng Gemini thật, không dùng fallback hoặc số ước lượng.

## Findings

1. **Biggest win:** child→parent làm recall tăng 0.2333 so với production trước sửa và khôi phục ô bảng/câu phủ định bị cắt.
2. **Hybrid có ý nghĩa:** BM25 mạnh với số/mã; dense nối paraphrase; RRF hợp nhất theo rank thay vì trộn score.
3. **Version là control point:** production tiếp theo nên chuẩn hóa `version`, `effective_date`, `status` để filter policy superseded.
4. **Multi-hop còn khó:** câu Senior + phép + lương cần hai tài liệu; query decomposition là fix tiếp theo.
5. **Độ bền API:** generation và RAGAS dùng model Gemini riêng, limiter 12 RPM; thiếu key/quota vẫn fallback an toàn.

## Bonus evidence

- Faithfulness ≥ 0.85: **0.9250** (+3).
- Tất cả metric ≥ 0.75: thấp nhất **0.7750** (+3).
- Enrichment combined 1 call/chunk: `_enrich_single_call()` (+2).
- Latency breakdown đo thật: `analysis/latency_breakdown.md` (+2).

## Demo ngắn

1. Chạy `pytest tests/ -v` và `python check_lab.py`.
2. So sánh hai JSON trong `reports/`.
3. Trình bày case bảng mua sắm bị child cắt và fix `parent_text`/`parent_id`.
4. Mở bottom-5 trong `analysis/failure_analysis.md` và suggested fix có regression test.
