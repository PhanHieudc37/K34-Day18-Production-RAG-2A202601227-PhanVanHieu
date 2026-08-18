# Failure Analysis — Lab 18: Production RAG

**Sinh viên:** Phan Văn Hiếu

**Ngày chạy:** 18/08/2026

**Evaluator:** RAGAS, Gemini 3.1 Flash Lite, 20 câu hỏi

## Kết quả tổng hợp

| Metric | Naive | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.3875 | **0.9250** | **+0.5375** |
| Answer Relevancy | 0.5435 | **0.8835** | **+0.3401** |
| Context Precision | 0.5250 | **0.7750** | **+0.2500** |
| Context Recall | 0.5750 | **0.9250** | **+0.3500** |

Production đạt cả bốn metric trên 0.75 và faithfulness trên 0.85. Thay đổi có ảnh hưởng lớn nhất là trả parent đầy đủ sau khi retrieve/rerank child. Trước sửa, bảng mua sắm bị cắt ngay trước ô “CEO” và câu PVI bị cắt sau từ “chưa”; sau sửa, context recall tăng từ 0.6917 lên 0.9250.

## Bottom-5 theo Error Tree

### 1. Senior 9 năm: phép và lương — average 0.3750

- **Expected:** 18 ngày phép; lương Senior 20–35 triệu/tháng.
- **Actual:** Trả đúng 18 ngày nhưng không tìm thấy bảng lương.
- **Worst metric:** Answer Relevancy = 0.0.
- **Error Tree:** Answer chưa đủ → context không có bằng chứng lương → retriever dành cả top-3 cho policy nghỉ phép → lỗi coverage ở M2/M3.
- **Diagnosis:** Query có hai intent thuộc hai tài liệu; top-k chung không bảo đảm mỗi intent có một nguồn.
- **Suggested fix:** Tách query thành “phép theo thâm niên” và “lương Senior”, RRF hai tập rồi rerank; test context phải chứa cả `nghi_phep_nam_v2024.md` và `bang_luong_2024.md`.

### 2. Tạm ứng 15 triệu quá hạn 5 ngày — average 0.6614

- **Expected/Actual:** Đều kết luận xấp xỉ 50.000 VNĐ.
- **Worst metric:** Faithfulness = 0.2.
- **Error Tree:** Answer đúng ground truth → context có 2%/tháng nhưng không nói rõ quy ước 30 ngày/pro-rata → generation thêm giả định tính toán.
- **Diagnosis:** Câu trả lời chưa gắn nhãn giả định.
- **Suggested fix:** Trả “xấp xỉ 50.000 VNĐ nếu quy ước tháng 30 ngày”; test không trình bày giả định như điều khoản policy.

### 3. Thâm niên được cộng phép — average 0.7187

- **Expected:** v2024 là 3 năm; v2023 cũ là 5 năm.
- **Actual:** Đúng policy hiện hành nhưng không nêu policy cũ.
- **Worst metric:** Context Precision = 0.0.
- **Error Tree:** Answer chính đúng → context chứa v2024 và v2023 → tài liệu superseded còn trong top-k → lỗi version filtering.
- **Diagnosis:** `source` được giữ nhưng `status=current|superseded` chưa dùng để filter.
- **Suggested fix:** Trích version/effective date/status, boost current policy; test v2024 đứng trước v2023.

### 4. Chu kỳ đổi mật khẩu — average 0.7193

- **Expected/Actual:** Đúng 120 ngày theo v2.0.
- **Worst metric:** Context Precision = 0.0.
- **Error Tree:** Answer đúng → context có v2 hiện hành và v1 cũ → một context không cần thiết → lỗi precision/version filtering.
- **Diagnosis:** Reranker chưa phạt mạnh metadata “ĐÃ THAY THẾ”.
- **Suggested fix:** Filter superseded khi đã có current policy; test top context chỉ chứa v2 nếu câu hỏi không yêu cầu lịch sử.

### 5. Phân loại thông tin lương — average 0.7251

- **Expected:** Bí mật, cấp 3, mã hóa và need-to-know.
- **Actual:** Đúng cấp 3 nhưng thiếu quy tắc xử lý.
- **Worst metric:** Context Precision = 0.0.
- **Error Tree:** Answer đúng nhưng thiếu ý → hai parent có bằng chứng → generation chưa tổng hợp đủ hai nguồn.
- **Diagnosis:** Prompt ưu tiên câu trả lời quá ngắn.
- **Suggested fix:** Trả schema `cấp độ + nhãn + quy tắc xử lý`; test yêu cầu cả “mã hóa” và “need-to-know”.

## Case study: child → parent

1. M2 tìm child nhỏ để có độ chính xác cao.
2. Phiên bản cũ trả child 256 ký tự, làm vỡ bảng/câu phủ định dù còn `parent_id`.
3. Phiên bản cuối giữ `parent_text`, loại child trùng parent trước rerank và trả parent sau rerank.
4. Trước → sau: faithfulness 0.9000 → 0.9250; relevancy 0.7582 → 0.8835; precision 0.6917 → 0.7750; recall 0.6917 → 0.9250.

## Ưu tiên vòng tiếp theo

1. Query decomposition cho câu hỏi đa-hop.
2. Metadata version/status và filter policy cũ.
3. Section-level parent/compression để giảm context thừa nhưng giữ nguyên bảng và ngoại lệ.
4. Regression tests cho phủ định, bảng Markdown, multi-document và version conflict.
