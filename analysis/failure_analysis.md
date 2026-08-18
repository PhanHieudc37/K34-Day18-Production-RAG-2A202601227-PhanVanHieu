# Failure Analysis — Lab 18: Production RAG

**Hình thức:** Cá nhân
**Sinh viên:** Phan Văn Hiếu
**Ngày chạy:** 18/08/2026

## Phạm vi đánh giá

Lần chạy nộp bài không có `OPENAI_API_KEY`, đồng thời hai model BGE chưa có trong cache. Pipeline đã chạy end-to-end bằng hashing dense fallback, lexical reranker fallback và RAGAS zero-score fallback. Do đó, các giá trị `0.0000` dưới đây có nghĩa là **chưa đo được bằng RAGAS**, không phải chất lượng thực bằng 0.

Để không tạo số liệu giả, bottom-5 được chọn bằng diagnostic pass trên chính 20 câu test: so sánh ground truth với top-3 context, kiểm tra nguồn/version và đọc top-1 answer fallback. Điểm overlap chỉ dùng để sắp ca cần đọc trước, không được báo cáo như RAGAS.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ | Trạng thái |
|---|---:|---:|---:|---|
| Faithfulness | 0.0000 | 0.0000 | +0.0000 | Không có API key |
| Answer Relevancy | 0.0000 | 0.0000 | +0.0000 | Không có API key |
| Context Precision | 0.0000 | 0.0000 | +0.0000 | Không có API key |
| Context Recall | 0.0000 | 0.0000 | +0.0000 | Không có API key |

## Bottom-5 Failures

### #1 — Mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt; cần xác nhận cấu hình từ CNTT và ít nhất 3 báo giá.
- **Got:** Top answer nói về điều kiện tài trợ đào tạo. Top-3 nguồn là `hoan_chi_dao_tao.md`, `dao_tao_noi_bo.md`, `thu_viec.md`.
- **Worst metric proxy:** Context Recall/Precision.
- **Error Tree:** Output sai → context không có bằng chứng mua sắm → query hợp lệ → retrieval xếp nhầm các chunk cùng chứa “30 triệu”, “nhân viên”, “phê duyệt”.
- **Root cause:** BM25/hashing dense ưu tiên overlap bề mặt; không có semantic BGE/reranker thật và chưa có query decomposition cho câu hỏi nhiều điều kiện.
- **Suggested fix:** Tách query thành `laptop 30 triệu phê duyệt`, `thiết bị CNTT xác nhận cấu hình`, `trên 10 triệu báo giá`; fusion theo sub-query và thêm regression test yêu cầu top-3 chứa `mua_sam.md`.

### #2 — Senior 9 năm: phép và lương

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép theo v2024 và lương Senior 20–35 triệu/tháng.
- **Got:** Top-1 là `nghi_phep_khong_luong.md`; top-2 có `nghi_phep_nam_v2024.md`, nhưng top-3 không có `bang_luong_2024.md`.
- **Worst metric proxy:** Context Recall.
- **Error Tree:** Output sai → context chỉ đủ một nửa câu hỏi → query gồm hai intent → retriever không bảo đảm coverage theo intent.
- **Root cause:** Một lượt top-k chung làm tài liệu nghỉ phép chiếm nhiều slot, còn bảng lương bị loại trước rerank.
- **Suggested fix:** Query decomposition thành hai nhánh “phép theo thâm niên” và “lương Senior”, lấy top-k mỗi nhánh rồi RRF; test nguồn cuối phải chứa cả `nghi_phep_nam_v2024.md` và `bang_luong_2024.md`.

### #3 — Lương thử việc Junior

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Top contexts có `thu_viec.md` (85%) và `bang_luong_2024.md` (mức Junior), nhưng fallback answer chỉ trả nguyên top-1 context nên không tính ra kết quả.
- **Worst metric proxy:** Answer Relevancy.
- **Error Tree:** Output chưa trả con số → context đủ bằng chứng ở nhiều chunk → retrieval chấp nhận được → generation/fallback không tổng hợp và tính toán.
- **Root cause:** Khi không có LLM, `run_query()` dùng `contexts[0]` làm answer; đây không phải một bước reasoning đa context.
- **Suggested fix:** Thêm deterministic calculator cho pattern phần trăm × mức lương hoặc yêu cầu prompt trích số từ mọi context và trình bày phép tính; test expected exact `17.000.000`.

### #4 — Thử việc có PVI không

- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Không; chỉ tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Top context bắt đầu bằng “được hưởng gói bảo hiểm sức khỏe PVI”, làm mất phần phủ định trước đó và có nguy cơ đảo ngược câu trả lời.
- **Worst metric proxy:** Faithfulness/Context Recall.
- **Error Tree:** Output có thể trái ground truth → context đã mất từ “chưa/không” → retrieval tìm đúng nguồn `thu_viec.md` → lỗi nằm ở child boundary M1.
- **Root cause:** Child được cắt theo giới hạn ký tự/whitespace, có thể tách giữa chủ ngữ + phủ định và vị ngữ.
- **Suggested fix:** Child splitter ưu tiên biên câu; nếu phải hard-split thì thêm overlap một câu hoặc 30–50 ký tự. Thêm unit test khẳng định cụm “chưa được hưởng gói bảo hiểm sức khỏe PVI” nằm trọn trong ít nhất một child.

### #5 — Phép năm hiện hành

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Chính sách v2024: 15 ngày; v2023 12 ngày đã bị thay thế.
- **Got:** Top-1 là `nghi_phep_khong_luong.md`, top-2 là nghỉ đặc biệt, top-3 là `nghi_phep_nam_v2023.md`; `nghi_phep_nam_v2024.md` không vào top-3.
- **Worst metric proxy:** Context Precision/Recall.
- **Error Tree:** Output sai phiên bản → context chứa policy cũ và policy khác loại → query hợp lệ → retrieval/rerank chưa dùng metadata hiệu lực.
- **Root cause:** Metadata hiện giữ được `source`, nhưng chưa chuẩn hóa `version`, `effective_date`, `status=current|superseded` để filter/boost.
- **Suggested fix:** Trích version/effective date ở M5, filter `status=current` trước RRF hoặc boost tài liệu mới; regression test yêu cầu v2024 đứng trên v2023 và nghỉ không lương.

## Case Study — Phép năm v2024/v2023

**Question chọn phân tích:** Nhân viên được nghỉ bao nhiêu ngày phép năm?

**Error Tree walkthrough:**

1. **Output đúng?** Không; top-1 nói về nghỉ không lương.
2. **Context đúng?** Không đủ; v2024 vắng mặt, v2023 xuất hiện ở hạng 3.
3. **Query rewrite đúng?** Query rõ, nhưng chưa bổ sung intent “chính sách hiện hành”.
4. **Fix ở bước:** M5 chuẩn hóa version/status → M2 filter/boost current policy → M3 rerank với source/version trong pair.
5. **Cách kiểm tra lại:** Test query phải đưa `nghi_phep_nam_v2024.md` lên hạng 1 và không dùng số 12 làm answer.

## Nếu có thêm 1 giờ

1. Sửa hierarchical child splitter theo biên câu + overlap để không làm rơi phủ định.
2. Thêm metadata `version`, `effective_date`, `status` và current-policy filter.
3. Thêm query decomposition cho câu hỏi cần tổng hợp nhiều tài liệu.
4. Tải model BGE và chạy lại RAGAS bằng API key cá nhân để thay toàn bộ zero fallback bằng số đo thật.
