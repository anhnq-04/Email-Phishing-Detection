# Phishing Detection System

Hệ thống phát hiện lừa đảo đa tầng (Multi-layer Phishing Detection System) — sử dụng kết hợp Rule-based, Machine Learning và Large Language Model.

## Kiến trúc hệ thống

```
                    ┌─────────────────┐
                    │   Input (Email/  │
                    │   URL / SMS)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Layer 1        │
                    │  Rule-Based     │
                    │  Fast Scan      │
                    └────┬───┬───┬────┘
                   PHISHING  │  SAFE
                         UNKNOWN
                             │
                    ┌────────▼────────┐
                    │  Layer 2        │
                    │  CatBoost ML    │
                    │  URL Classifier │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Layer 3        │
                    │  Qwen LLM       │
                    │  Content Analysis│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Verdict   │
                    └─────────────────┘
```

### Layer 1 — Rule-Based Fast Scan
- Xác thực SPF/DKIM headers
- Kiểm tra blacklist domains
- Phát hiện mạo danh thương hiệu (NER + whitelist)
- Phát hiện tấn công homograph (Unicode)

### Layer 2 — CatBoost URL Classifier
- Trích xuất 15 features từ URL và HTML
- Phân loại URL bằng CatBoost model
- Features: IsHTTPS, DomainLength, NoOfSubDomain, NoOfJS, ...

### Layer 3 — Qwen LLM Content Analysis
- Fine-tuned Qwen model trên dataset email/SMS tiếng Việt & tiếng Anh
- Phân tích nội dung sâu để phát hiện phishing tinh vi

## Cấu trúc thư mục

```
├── configs/           # Cấu hình YAML (model, training, inference)
├── data/
│   ├── raw/           # Dữ liệu thô (CSV, JSONL, lists)
│   └── processed/     # Dữ liệu đã xử lý
├── docs/              # Tài liệu
├── frontend/          # Giao diện web
├── logs/              # Log files
├── models/            # Model weights (CatBoost .cbm, Qwen adapter)
├── notebooks/         # Jupyter notebooks (EDA, experiments)
├── scripts/           # Scripts tiện ích (test, crawl, data gen)
├── src/               # Source code chính
│   ├── api/           # FastAPI endpoints
│   ├── inference/     # Pipeline điều phối
│   ├── models/        # Layer 1, 2, 3
│   ├── preprocessing/ # Tiền xử lý dữ liệu
│   ├── training/      # Huấn luyện model
│   └── utils/         # Tiện ích dùng chung
└── requirements.txt
```

## Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd Phishing-Detection-System

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 🔧 Sử dụng

### Chạy API Server
```bash
uvicorn src.api.app:app --reload --port 8000
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/qwen3-14b-unsloth-bnb-4bit \
  --enable-lora \
  --lora-modules phishing=models/qwen/model_final \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 4096 \
  --port 8001 \
  --gpu-memory-utilization 0.9
```

### Phân tích URL
```python
from src.models.catboost.layer2 import predict_url

result = predict_url("https://example.com")
print(result)
```

### Phân tích Email
```python
from src.inference.pipeline import PhishingDetectionPipeline

pipeline = PhishingDetectionPipeline()
result = pipeline.analyze_email(
    sender_email="support@example.com",
    body_text="Nội dung email...",
)
print(result)
```

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/analyze/email` | Phân tích email |
| POST | `/api/v1/analyze/url` | Phân tích URL |

## 📝 License

This project is for academic purposes.
