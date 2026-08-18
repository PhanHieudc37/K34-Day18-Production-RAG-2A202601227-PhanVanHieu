# Latency Breakdown — Production RAG

- Provider: **gemini**
- Evaluation queries: **20**
- Measurement: `time.time()` wall-clock, measured by the submission pipeline.
- Model downloads are excluded; reranker/model loading is reported separately.

| Stage | Calls | Total (s) | Average (ms/call) |
|---|---:|---:|---:|
| M1 chunking | 1 | 0.143 | 142.87 |
| M5 enrichment | 1 | 243.659 | 243659.45 |
| M2 BM25 + dense indexing | 1 | 145.290 | 145289.68 |
| M3 reranker load | 1 | 0.001 | 1.14 |
| M2 hybrid retrieval | 20 | 60.633 | 3031.66 |
| M3 reranking | 20 | 872.319 | 43615.96 |
| LLM answer generation (final parent contexts, rate-limited) | 20 | 189.700 | 9485.00 |
| M4 RAGAS evaluation (final clean run) | 1 | 645.200 | 645200.00 |
| **Measured total** | — | **2156.945** | — |

## CPU/model observations

- `BAAI/bge-m3` cache: khoảng 4.25 GB; load offline 3.758 s; encode 2 câu 2.160 s; vector đúng 1024 chiều.
- `BAAI/bge-reranker-v2-m3` cache: khoảng 2.14 GB; cold run 8.763 s; warm benchmark 5 documents trung bình 1.010 s.
- Full reranking 20 candidates/query là bottleneck lớn nhất trên CPU (43.616 s/query). Production thực tế nên batch, dùng GPU hoặc reranker nhỏ hơn nếu SLA nghiêm ngặt.
- Gemini được giới hạn 12 RPM để nằm dưới free-tier 15 RPM; latency cao hơn nhưng loại bỏ metric 0 giả do HTTP 429.
