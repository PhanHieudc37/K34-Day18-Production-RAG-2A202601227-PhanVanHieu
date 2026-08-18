# Individual Reflection — Lab 18

**Tên:** Phan Văn Hiếu
**Module phụ trách:** M1–M5 (bài làm cá nhân)
**Ngày:** 18/08/2026

## 1. Đóng góp kỹ thuật

- **M1 — Chunking:** implement `chunk_semantic()`, `chunk_hierarchical()` và `chunk_structure_aware()`. Semantic dùng cosine similarity và cache model; hierarchical tạo stable `parent_id`; structure-aware giữ heading, table, list và code fence.
- **M2 — Search:** implement tokenizer tiếng Việt, BM25, Qdrant dense search và Reciprocal Rank Fusion. Payload giữ `text`, `source`, version metadata và parent link.
- **M3 — Reranking:** implement `CrossEncoderReranker` bằng `sentence_transformers.CrossEncoder`, sort score giảm dần, giữ original score/metadata/rank và có fallback offline.
- **M4 — Evaluation:** implement RAGAS bốn metric, chuyển row thành `EvalResult`, zero fallback không crash và Diagnostic Tree cho bottom failures.
- **M5 — Enrichment:** implement summary, HyQA, contextual prepend, metadata extraction và combined one-call mode; metadata gốc có precedence để không mất source/version.
- Sửa `main.py` dùng `os.replace()` để report cũ được ghi đè an toàn trên Windows.
- Kết quả kiểm thử: **37/37 tests pass**; không còn TODO trong M1–M5.

## 2. Mapping concept → code

| Concept | Hiện thực cụ thể | Điều rút ra |
|---|---|---|
| Semantic chunking | `chunk_semantic()` | Threshold chỉ hữu ích khi model sẵn sàng; cần fallback và cache để pipeline không treo offline. |
| Hierarchical chunking | `chunk_hierarchical()` | Parent giúp giữ context, nhưng child boundary phải bảo toàn câu và từ phủ định. |
| Structure-aware chunking | `chunk_structure_aware()` | Heading breadcrumb và code-fence state tránh phá cấu trúc Markdown. |
| BM25 tiếng Việt | `segment_vietnamese()`, `BM25Search` | Phải đổi `_` thành khoảng trắng đúng contract; keyword search mạnh với số, policy name và mã. |
| Dense + Qdrant | `DenseSearch` | Vector dimension/payload là contract; metadata phải đi cùng vector để debug nguồn/version. |
| RRF | `reciprocal_rank_fusion()` | Chỉ fusion theo rank vì BM25 score và cosine score không cùng thang đo. |
| Cross-encoder | `CrossEncoderReranker.rerank()` | Rerank ít ứng viên để đổi latency lấy precision; cần giữ pre-rerank score để chẩn đoán. |
| RAGAS | `evaluate_ragas()` | Bốn metric tách lỗi answer khỏi lỗi context; fallback 0 phải được ghi là “unavailable”, không diễn giải là quality score. |
| Diagnostic Tree | `failure_analysis()` | Bottom-N chỉ hữu ích khi gắn metric tệ nhất với một fix có thể test lại. |
| Combined enrichment | `_enrich_single_call()`, `enrich_chunks()` | Một call/chunk kiểm soát cost; original metadata phải thắng metadata do LLM sinh. |

## 3. Kiến thức học được

- **Khái niệm mới nhất:** RRF giải quyết vấn đề hợp nhất hai ranking mà không cần hiệu chỉnh score giữa lexical và vector search.
- **Điều bất ngờ nhất:** Một child chunk bắt đầu ở “được hưởng PVI” nhưng làm rơi từ “chưa” có thể đảo nghĩa hoàn toàn dù source retrieval đúng.
- **Kết nối với bài giảng:** Chất lượng RAG là chuỗi contract. Model generation không thể sửa context đã mất phủ định, sai version hoặc thiếu một nửa bằng chứng multi-hop.
- **Version awareness:** Lưu `source` là cần nhưng chưa đủ; production cần `effective_date`, `version`, `status=current|superseded` và filter rõ ràng.

## 4. Khó khăn & cách giải quyết

- Máy ban đầu chỉ nhận Python 3.10; dùng `uv` tạo `.venv` Python 3.11.15.
- `pip` gặp resolver cũ, timeout và file wheel bị khóa; nâng pip rồi dùng cache/download của `uv`.
- Docker registry TLS timeout; dùng image Qdrant 1.18.3 đã có local và xác nhận HTTP port 6333.
- PowerShell cp1252 làm lỗi emoji/tiếng Việt; chạy Python với `PYTHONUTF8=1` và `PYTHONIOENCODING=utf-8`.
- BGE-M3 và reranker chưa có cache; thêm fallback deterministic, chạy model hub offline và ghi rõ giới hạn benchmark.
- `main.py` thất bại ở lần chạy lặp lại vì `os.rename()` không overwrite trên Windows; đổi sang `os.replace()`.
- **Thời gian debug:** Không đo riêng từng lỗi; phần lớn thời gian nằm ở mạng/model cache và kiểm tra end-to-end trên Windows.

## 5. Nếu làm lại

- Tạo test offline/fake encoder từ đầu để tách lỗi contract khỏi lỗi tải model.
- Pin phiên bản dependency thay vì chỉ dùng lower/upper bounds rộng, đặc biệt `sentence-transformers`, `transformers`, `openai` và `qdrant-client`.
- Thiết kế metadata schema cho version ngay ở M1/M5 trước khi index.
- Dùng sentence-boundary + overlap cho child chunks thay vì chỉ giới hạn ký tự.
- Lưu per-query retrieval trace (BM25 rank, dense rank, RRF rank, rerank score) thành artifact để failure analysis tái lập được.

## 6. Action plan áp dụng vào project

1. Thêm unit test không làm rơi negation ở child boundary.
2. Thêm version-aware filter và test v2024 đứng trên v2023.
3. Thêm query decomposition cho câu hỏi cần hai nguồn.
4. Tải model thật, đo latency cold/warm và chạy lại RAGAS khi có API key cá nhân.
5. Chỉ cập nhật bảng metric/report sau khi có lần eval thật; không thay fallback bằng số ước lượng.

## 7. Tự đánh giá

| Tiêu chí | Tự chấm (1–5) | Lý do |
|---|---:|---|
| Hiểu bài giảng | 5 | Map được M1–M5 vào failure thực tế và version conflict. |
| Code quality | 4 | Có type contract, fallback, metadata preservation và test; còn cần model benchmark thật. |
| Teamwork/ownership | 5 | Hoàn thành toàn bộ module trong bài cá nhân và ghi rõ giới hạn. |
| Problem solving | 5 | Xử lý môi trường, network, Docker, encoding, model fallback và Windows report move. |
