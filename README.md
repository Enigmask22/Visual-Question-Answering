# VQA System — Visual Question Answering with Multi-Architecture Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/transformers-4.20+-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Gradio-4.0+-orange.svg" alt="Gradio">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED.svg" alt="Docker">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

> An end-to-end **Visual Question Answering** system that answers natural language questions about images
> by fusing visual and textual understanding. Ships with **3 model architectures**, a **production inference engine**,
> an **interactive Gradio web demo**, a **FastAPI REST API**, and **Docker deployment** — ready for both research experiments and real-world serving.

---

## Highlights

- **3 Model Architectures** — from simple MLP baseline to state-of-the-art ViT+BERT, enabling systematic benchmarking
- **Vietnamese Language Support** — ViT+BERT uses VinAI's PhoBERT for bilingual (EN/VI) question answering
- **Production Inference Engine** — single & batch prediction, GPU/CPU auto-detection, confidence scoring, sub-second response
- **Gradio Web Demo** — upload an image, ask a question, get Top-K answers instantly in the browser
- **FastAPI REST API** — production-grade endpoints with Swagger/ReDoc docs, input validation, health checks
- **Docker Ready** — multi-service Docker Compose with CUDA GPU passthrough, health checks, volume mounts
- **Config-Driven** — switch models, hyperparameters, datasets via JSON config without touching code
- **5 Evaluation Metrics** — Top-1/Top-5 Accuracy, VQA Accuracy, and ANLS following standard benchmarking protocols

---

## Architecture Overview

| Model            | Image Encoder                         | Text Encoder              | Fusion             | Use Case                             |
| :--------------- | :------------------------------------ | :------------------------ | :----------------- | :----------------------------------- |
| **MLP Baseline** | Flattened Pixels                      | TF-IDF                    | Concatenation + FC | Quick baseline, CPU-friendly         |
| **CNN + LSTM**   | ResNet50 (pretrained, frozen)         | Bi-LSTM + Word Embeddings | Bottleneck fusion  | Deep learning traditional            |
| **ViT + BERT**   | Vision Transformer (ViT-B/16, frozen) | PhoBERT (frozen)          | Bottleneck fusion  | State-of-the-art, Vietnamese support |

**Pipeline**: Image → Encoder → Visual Features ⊕ Text Features → Bottleneck Fusion → Classifier → Top-K Answers

**Datasets supported**: VQAv2 (English) and ViInfographicVQA (Vietnamese).

---

## Project Structure

```
Visual_Question_Answering/
├── train.py                    # Model training entry point
├── test.py                     # Model evaluation entry point
├── inference.py                # Production inference engine (VQAInferenceEngine)
├── app.py                      # Gradio interactive web application
├── api.py                      # FastAPI REST API server
├── parse_config.py             # Config parser with CLI override support
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Multi-stage Docker build (CUDA support)
├── docker-compose.yml          # Multi-service orchestration (API + UI)
│
├── config/                     # JSON configuration files
│   ├── config_mlp.json                     # MLP on VQAv2
│   ├── config_cnn_lstm.json                # CNN+LSTM on VQAv2
│   ├── config_vit_bert.json                # ViT+BERT on VQAv2
│   ├── config_mlp_viinfographic.json       # MLP on ViInfographicVQA
│   ├── config_cnn_lstm_viinfographic.json  # CNN+LSTM on ViInfographicVQA
│   └── config_vit_bert_viinfographic.json  # ViT+BERT on ViInfographicVQA
│
├── model/                      # Model architectures
│   ├── mlp.py                  # MLPBaseline (TF-IDF + Flattened Image)
│   ├── cnn_lstm.py             # CNNLSTMBaseline (ResNet50 + Bi-LSTM)
│   ├── vit_bert.py             # ViTBERTBaseline (ViT + PhoBERT)
│   ├── loss.py                 # Loss functions (CrossEntropy, NLL)
│   └── metrics.py              # VQA Accuracy, ANLS, Top-K Accuracy
│
├── data_loader/                # Data loading utilities
│   ├── dataset.py              # VQADataset (PyTorch Dataset)
│   └── data_loaders.py         # Vocab builder, transforms, DataLoader
│
├── trainer/                    # Training logic
│   └── trainer.py              # VQATrainer with validation & TensorBoard
│
├── base/                       # Abstract base classes
│   ├── base_model.py           # BaseModel (nn.Module with param counting)
│   ├── base_trainer.py         # BaseTrainer (checkpointing, early stopping)
│   └── base_data_loader.py     # BaseDataLoader
│
├── logger/                     # Logging & visualization
│   ├── logger.py               # Logger setup
│   └── visualization.py        # TensorBoard writer
│
├── utils/                      # Utilities
│   └── util.py                 # JSON I/O, device setup, MetricTracker
│
├── data/                       # Dataset directory (gitignored)
│   ├── vqav2/
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── test.json
│   │   └── images/
│   └── viinfographic/
│       ├── train.json
│       ├── val.json
│       ├── test.json
│       └── images/
│
└── saved/                      # Checkpoints & logs (gitignored)
    ├── models/
    └── log/
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- CUDA GPU recommended (auto-fallback to CPU)
- ~4 GB disk for pretrained weights (ViT, ResNet50, PhoBERT)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Enigmask22/Visual_Question_Answering.git
cd Visual_Question_Answering

# Create virtual environment
python -m venv venv
# Linux/Mac: source venv/bin/activate
# Windows:   .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Structure your data in the `data/` directory:

```
data/vqav2/
├── train.json      # [{"image_path": "images/xxx.jpg", "question": "...", "answer": "...", "answers": ["...", ...]}, ...]
├── val.json
├── test.json
└── images/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

Each JSON entry must have:
| Field | Type | Description |
|:------|:-----|:------------|
| `image_path` | `str` | Relative path to image from data dir |
| `question` | `str` | Natural language question |
| `answer` | `str` | Primary answer |
| `answers` | `list[str]` | Multiple annotator answers (for VQA Accuracy) |

### 3. Train a Model

```bash
# Train MLP Baseline (fastest, good for testing pipeline)
python train.py -c config/config_mlp.json

# Train CNN + LSTM (ResNet50 + Bi-LSTM)
python train.py -c config/config_cnn_lstm.json

# Train ViT + BERT (best accuracy, requires more VRAM)
python train.py -c config/config_vit_bert.json

# Train on Vietnamese ViInfographicVQA dataset
python train.py -c config/config_vit_bert_viinfographic.json

# Override hyperparameters via CLI
python train.py -c config/config_cnn_lstm.json --lr 0.0001 --bs 64

# Select GPU device
python train.py -c config/config_vit_bert.json --device 0

# Resume from checkpoint
python train.py -c config/config_cnn_lstm.json -r saved/models/VQA_CNN_LSTM/MMDD_HHMMSS/checkpoint-epoch5.pth
```

**Monitor training** with TensorBoard:

```bash
tensorboard --logdir saved/log/
# Open http://localhost:6006 for loss curves, learning rate schedules, and metrics
```

### 4. Evaluate

```bash
# Evaluate on test set
python test.py -c config/config_vit_bert.json -r saved/models/VQA_ViT_BERT/MMDD_HHMMSS/model_best.pth

# Custom batch size for evaluation
python test.py -c config/config_cnn_lstm.json -r saved/models/VQA_CNN_LSTM/MMDD_HHMMSS/model_best.pth --bs 64
```

**Evaluation metrics reported:**

| Metric             | Description                                   |
| :----------------- | :-------------------------------------------- |
| Top-1 Accuracy     | Exact match with ground truth label           |
| Top-5 Accuracy     | Ground truth in top-5 predictions             |
| VQA Accuracy       | Multi-annotator consensus (min(count/3, 1.0)) |
| VQA Top-5 Accuracy | VQA accuracy over top-5 predictions           |
| ANLS               | Average Normalized Levenshtein Similarity     |

---

## Serving & Deployment

### Option A: Gradio Web Demo

Interactive browser-based demo — upload image, type question, get answers.

```bash
python app.py \
    -c config/config_vit_bert.json \
    -r saved/models/VQA_ViT_BERT/MMDD_HHMMSS/model_best.pth \
    --port 7860
```

Open `http://localhost:7860` in your browser.

**Features:**

- Drag & drop image upload
- Real-time Top-K answer ranking with confidence scores
- Model architecture info display
- Enter key shortcut for prediction
- Optional `--share` flag for public Gradio link

```bash
# Create a public shareable link
python app.py -c config/config_vit_bert.json -r saved/models/model_best.pth --share
```

### Option B: FastAPI REST API

Production-grade REST API with automatic Swagger documentation.

```bash
python api.py \
    -c config/config_vit_bert.json \
    -r saved/models/VQA_ViT_BERT/MMDD_HHMMSS/model_best.pth \
    --host 0.0.0.0 \
    --port 8000
```

**Endpoints:**

| Method | Endpoint       | Description                      |
| :----- | :------------- | :------------------------------- |
| `GET`  | `/health`      | Health check & model status      |
| `GET`  | `/model/info`  | Model architecture details       |
| `POST` | `/predict`     | Predict from uploaded image file |
| `POST` | `/predict/url` | Predict from local image path    |

**Interactive docs:** `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc).

**Example — cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
    -F "image=@photo.jpg" \
    -F "question=What color is the car?" \
    -F "top_k=5"
```

**Example — Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    files={"image": open("photo.jpg", "rb")},
    data={"question": "What color is the car?", "top_k": 5},
)
result = response.json()
print(f"Answer: {result['answer']} ({result['confidence']}%)")
print(f"Inference time: {result['inference_time_ms']}ms")
```

**Example — Response:**

```json
{
  "answer": "red",
  "confidence": 87.32,
  "top_k_answers": [
    { "answer": "red", "confidence": 87.32, "rank": 1 },
    { "answer": "blue", "confidence": 5.21, "rank": 2 },
    { "answer": "white", "confidence": 3.47, "rank": 3 }
  ],
  "inference_time_ms": 142.5,
  "model_type": "ViTBERTBaseline",
  "question": "What color is the car?"
}
```

### Option C: CLI Inference

Quick single-image inference from the command line.

```bash
python inference.py \
    -c config/config_vit_bert.json \
    -r saved/models/model_best.pth \
    -i data/vqav2/images/example.jpg \
    -q "What is in the image?" \
    --top_k 5
```

### Option D: Docker Deployment

Deploy both API and Web UI with Docker Compose.

```bash
# Build and start all services
docker compose up --build -d

# API server:  http://localhost:8000  (Swagger: /docs)
# Gradio UI:   http://localhost:7860
```

**Services:**

| Service   | Container | Port | Description                         |
| :-------- | :-------- | :--- | :---------------------------------- |
| `vqa-api` | vqa-api   | 8000 | FastAPI REST API with health checks |
| `vqa-ui`  | vqa-ui    | 7860 | Gradio interactive web demo         |

Both services mount `./data` and `./saved` as volumes, support NVIDIA GPU passthrough, and auto-restart on failure.

```bash
# Check service health
curl http://localhost:8000/health

# View logs
docker compose logs -f vqa-api

# Stop services
docker compose down
```

**CPU-only deployment** — remove the `deploy.resources.reservations` block and `CUDA_VISIBLE_DEVICES` from `docker-compose.yml`.

---

## Inference Engine API

The `VQAInferenceEngine` class in `inference.py` can be imported and used programmatically:

```python
from inference import VQAInferenceEngine

# Initialize engine
engine = VQAInferenceEngine(
    config_path="config/config_vit_bert.json",
    checkpoint_path="saved/models/model_best.pth",
    device="cuda",  # or "cpu", or None for auto-detection
)

# Single prediction
result = engine.predict(
    image="path/to/image.jpg",       # str, Path, PIL.Image, or numpy array
    question="What is in the image?",
    top_k=5,
)
print(result["answer"], result["confidence"])

# Batch prediction
results = engine.predict_batch(
    images=["img1.jpg", "img2.jpg"],
    questions=["What color?", "How many people?"],
    top_k=3,
)

# Model info
info = engine.get_model_info()
# {'model_type': 'ViTBERTBaseline', 'trainable_parameters': 1052672, ...}
```

---

## Configuration

All training/inference settings are controlled via JSON config files in `config/`. Key parameters:

```jsonc
{
  "name": "VQA_ViT_BERT", // Experiment name
  "n_gpu": 1, // Number of GPUs
  "arch": {
    "type": "ViTBERTBaseline", // Model: MLPBaseline | CNNLSTMBaseline | ViTBERTBaseline
    "args": {
      "bert_model": "vinai/phobert-base", // BERT model (ViT+BERT only)
      "hidden_dim": 512, // Fusion layer hidden dim
      "img_size": 224, // Input image size
      "max_len": 64, // Max question token length
    },
  },
  "data_loader": {
    "args": {
      "data_dir": "data/vqav2", // Dataset directory
      "batch_size": 32,
      "num_workers": 4,
      "img_size": 224,
    },
  },
  "optimizer": { "type": "Adam", "args": { "lr": 0.001 } },
  "loss": "cross_entropy_loss",
  "lr_scheduler": {
    "type": "StepLR",
    "args": { "step_size": 50, "gamma": 0.1 },
  },
  "trainer": {
    "epochs": 10,
    "monitor": "max val_vqa_accuracy", // Early stopping metric
    "early_stop": 10,
    "tensorboard": true,
  },
}
```

**Available configs:**

| Config File                          | Model    | Dataset          | Language   |
| :----------------------------------- | :------- | :--------------- | :--------- |
| `config_mlp.json`                    | MLP      | VQAv2            | English    |
| `config_cnn_lstm.json`               | CNN+LSTM | VQAv2            | English    |
| `config_vit_bert.json`               | ViT+BERT | VQAv2            | English    |
| `config_mlp_viinfographic.json`      | MLP      | ViInfographicVQA | Vietnamese |
| `config_cnn_lstm_viinfographic.json` | CNN+LSTM | ViInfographicVQA | Vietnamese |
| `config_vit_bert_viinfographic.json` | ViT+BERT | ViInfographicVQA | Vietnamese |

---

## Module Overview

| Module                        | Description                                                                                     |
| :---------------------------- | :---------------------------------------------------------------------------------------------- |
| `train.py`                    | Training entry point — loads config, builds model/dataloader/optimizer, launches `VQATrainer`   |
| `test.py`                     | Evaluation entry point — loads checkpoint, computes all 5 metrics on test set                   |
| `inference.py`                | `VQAInferenceEngine` — production inference with auto device, batch support, confidence scoring |
| `app.py`                      | Gradio web application — interactive image upload + Q&A with Top-K display                      |
| `api.py`                      | FastAPI REST API — `/predict`, `/health`, `/model/info` with Swagger docs                       |
| `parse_config.py`             | Config parser with CLI override (e.g., `--lr`, `--bs`), checkpoint resume support               |
| `model/mlp.py`                | `MLPBaseline` — Flattened pixels + TF-IDF → FC classifier                                       |
| `model/cnn_lstm.py`           | `CNNLSTMBaseline` — ResNet50 (frozen) + Bi-LSTM → Bottleneck → Classifier                       |
| `model/vit_bert.py`           | `ViTBERTBaseline` — ViT-B/16 (frozen) + PhoBERT (frozen) → Bottleneck → Classifier              |
| `model/metrics.py`            | `compute_vqa_accuracy`, `compute_anls`, `accuracy_top1`, `accuracy_top5`                        |
| `model/loss.py`               | `cross_entropy_loss`, `nll_loss`                                                                |
| `data_loader/dataset.py`      | `VQADataset` — PyTorch Dataset for VQA JSON format                                              |
| `data_loader/data_loaders.py` | `build_answer_vocab`, `get_transforms`, `VQADataLoader`                                         |
| `trainer/trainer.py`          | `VQATrainer` — training loop with validation, TensorBoard logging, metric tracking              |
| `base/base_trainer.py`        | `BaseTrainer` — checkpointing, early stopping, learning rate scheduling                         |
| `base/base_model.py`          | `BaseModel` — nn.Module with trainable parameter counting                                       |
| `utils/util.py`               | `prepare_device`, `MetricTracker`, `read_json`, `write_json`                                    |

---

## Tech Stack

| Category          | Technologies                                                       |
| :---------------- | :----------------------------------------------------------------- |
| **Deep Learning** | PyTorch, torchvision, timm (ViT, ResNet50), Transformers (PhoBERT) |
| **ML/NLP**        | Scikit-learn (TF-IDF), NLTK-style tokenization, Word Embeddings    |
| **Serving**       | FastAPI, Uvicorn, Gradio, Pydantic                                 |
| **DevOps**        | Docker, Docker Compose, CUDA, TensorBoard                          |
| **Data**          | Pillow, NumPy, Pandas, tqdm                                        |

---

## License

This project is licensed under the MIT License.
