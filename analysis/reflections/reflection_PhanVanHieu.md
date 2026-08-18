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
- **Production handoff:** retrieve/rerank child nhưng trả `parent_text`, đồng thời deduplicate theo `parent_id`; fix này đưa RAGAS production lên **0.9250 / 0.8835 / 0.7750 / 0.9250**.
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
- **Điều bất ngờ nhất:** Một child chunk bắt đầu ở “được hưởng PVI” nhưng làm rơi từ “chưa” có thể đảo nghĩa hoàn toàn dù source retrieval đúng. Giữ `parent_id` thôi chưa đủ; runtime phải thực sự resolve child về parent trước generation.
- **Kết nối với bài giảng:** Chất lượng RAG là chuỗi contract. Model generation không thể sửa context đã mất phủ định, sai version hoặc thiếu một nửa bằng chứng multi-hop.
- **Version awareness:** Lưu `source` là cần nhưng chưa đủ; production cần `effective_date`, `version`, `status=current|superseded` và filter rõ ràng.

## 4. Khó khăn & cách giải quyết

- Máy ban đầu chỉ nhận Python 3.10; dùng `uv` tạo `.venv` Python 3.11.15.
- `pip` gặp resolver cũ, timeout và file wheel bị khóa; nâng pip rồi dùng cache/download của `uv`.
- Docker registry TLS timeout; dùng image Qdrant 1.18.3 đã có local và xác nhận HTTP port 6333.
- PowerShell cp1252 làm lỗi emoji/tiếng Việt; chạy Python với `PYTHONUTF8=1` và `PYTHONIOENCODING=utf-8`.
- BGE-M3 và reranker ban đầu chưa có cache; thêm fallback deterministic, sau đó tải đủ hai model và xác minh offline. BGE-M3 encode đúng vector 1024 chiều; reranker cold 8.76 giây, warm khoảng 1.01 giây/5 documents trên CPU.
- OpenAI key xác thực nhưng completion trả chính xác `429 insufficient_quota`. Tôi thêm provider Gemini qua OpenAI-compatible endpoint. Lần đầu chạy Gemini gặp `429 RESOURCE_EXHAUSTED`: giới hạn 15 RPM và 500 request/ngày/model. Cách xử lý là dùng limiter 12 RPM, tách model generation (`gemini-3.5-flash-lite`) khỏi model eval (`gemini-3.1-flash-lite`), cache dataset trước RAGAS và chỉ rerun M4 khi cần.
- RAGAS Gemini không hỗ trợ `n > 1`; `answer_relevancy` mặc định strictness 3 tạo lỗi. Tôi đặt strictness 1 cho Gemini và xác minh lại đủ 20 câu, không dùng zero fallback.
- Diagnostic thật phát hiện hierarchical pipeline chỉ trả child 256 ký tự. Tôi giữ `parent_text` trong metadata, deduplicate candidate theo parent và trả parent sau rerank; recall tăng từ 0.6917 lên 0.9250.
- `main.py` thất bại ở lần chạy lặp lại vì `os.rename()` không overwrite trên Windows; đổi sang `os.replace()`.
- **Thời gian debug:** Không đo riêng từng lỗi; phần lớn thời gian nằm ở mạng/model cache và kiểm tra end-to-end trên Windows.

## 5. Nếu làm lại

- Tạo test offline/fake encoder từ đầu để tách lỗi contract khỏi lỗi tải model.
- Pin phiên bản dependency thay vì chỉ dùng lower/upper bounds rộng, đặc biệt `sentence-transformers`, `transformers`, `openai` và `qdrant-client`.
- Thiết kế metadata schema cho version ngay ở M1/M5 trước khi index.
- Dùng sentence-boundary + overlap cho child chunks thay vì chỉ giới hạn ký tự.
- Lưu per-query retrieval trace (BM25 rank, dense rank, RRF rank, rerank score) thành artifact để failure analysis tái lập được.

## 6. Action plan áp dụng vào project

Áp dụng trực tiếp vào project trợ lý tra cứu chính sách nhân sự cá nhân theo timeline sau:

1. **Tuần 1:** thêm unit test không làm rơi negation ở child boundary và sửa child split theo sentence boundary + overlap; tiêu chí đạt là toàn bộ test cũ và test negation mới đều pass.
2. **Tuần 2:** chuẩn hoá metadata `version`, `effective_date`, `status` và thêm version-aware filter; kiểm tra bằng bộ query bắt buộc policy v2024 đứng trên v2023.
3. **Tuần 3:** thêm query decomposition cho câu hỏi cần hai nguồn và lưu retrieval trace BM25/dense/RRF/rerank; kiểm tra bằng các câu phép + lương trong test set.
4. **Tuần 4:** đo latency cold/warm, cấu hình rate limit theo quota provider và chạy RAGAS định kỳ từ cached eval dataset. Gate phát hành: ít nhất 3 metric ≥ 0.70, faithfulness ≥ 0.85.
5. **Cuối tuần 4:** phân tích bottom-5 theo Error Tree, áp dụng một fix cho mỗi nhóm lỗi rồi regression eval. Lần lab này đã vượt gate với cả bốn metric ≥ 0.75 và faithfulness 0.9250.

## 7. Tự đánh giá

| Tiêu chí | Tự chấm (1–5) | Lý do |
|---|---:|---|
| Hiểu bài giảng | 5 | Map được M1–M5 vào failure thực tế và version conflict. |
| Code quality | 5 | Có contract, fallback, metadata preservation, API rate limit, cached eval, 37 tests và RAGAS thật đủ 20 câu. |
| Teamwork/ownership | 5 | Hoàn thành toàn bộ module trong bài cá nhân và ghi rõ giới hạn. |
| Problem solving | 5 | Xử lý môi trường, network, Docker, encoding, model fallback và Windows report move. |
