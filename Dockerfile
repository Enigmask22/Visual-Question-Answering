# ==========================================
# Stage 1: Base image với CUDA support
# ==========================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

WORKDIR /app

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Application
# ==========================================
FROM base AS app

WORKDIR /app

# Sao chép toàn bộ source code
COPY . .

# Tạo thư mục cần thiết
RUN mkdir -p saved/models saved/log data

# Expose ports: 8000 (API), 7860 (Gradio)
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Biến môi trường
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: chạy FastAPI server
CMD ["python", "api.py", \
     "-c", "config/config_vit_bert.json", \
     "-r", "saved/models/model_best.pth", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
