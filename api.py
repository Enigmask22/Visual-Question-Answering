"""
FastAPI REST API cho Visual Question Answering.
Cung cấp endpoint RESTful để tích hợp hệ thống,
hỗ trợ single/batch inference, upload ảnh, và health check.
"""

import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference import VQAInferenceEngine

logger = logging.getLogger(__name__)

# ======================== Pydantic Schemas ========================


class PredictionResponse(BaseModel):
    """Schema response cho dự đoán đơn."""
    answer: str = Field(..., description="Câu trả lời tốt nhất")
    confidence: float = Field(..., description="Độ tin cậy (%)")
    top_k_answers: List[Dict[str, Any]] = Field(
        ..., description="Danh sách top-k câu trả lời"
    )
    inference_time_ms: float = Field(..., description="Thời gian suy luận (ms)")
    model_type: str = Field(..., description="Kiến trúc model")
    question: str = Field(..., description="Câu hỏi đầu vào")


class BatchPredictionRequest(BaseModel):
    """Schema request cho dự đoán batch (sử dụng URL ảnh)."""
    image_urls: List[str] = Field(..., description="Danh sách URL ảnh")
    questions: List[str] = Field(..., description="Danh sách câu hỏi")
    top_k: int = Field(default=5, ge=1, le=20, description="Số câu trả lời top-k")


class HealthResponse(BaseModel):
    """Schema response cho health check."""
    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None


class ModelInfoResponse(BaseModel):
    """Schema response cho thông tin model."""
    model_type: str
    total_parameters: int
    trainable_parameters: int
    num_answer_classes: int
    image_size: int
    device: str
    checkpoint: str


# ======================== Application Factory ========================


def create_app(
    config_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
) -> FastAPI:
    """
    Factory function tạo FastAPI application.

    Args:
        config_path: Đường dẫn config JSON.
        checkpoint_path: Đường dẫn checkpoint .pth.
        device: Thiết bị tính toán.

    Returns:
        FastAPI application instance.
    """
    app = FastAPI(
        title="VQA - Visual Question Answering API",
        description=(
            "REST API cho hệ thống trả lời câu hỏi về ảnh (Visual Question Answering). "
            "Hỗ trợ 3 kiến trúc: MLP, CNN+LSTM (ResNet50), ViT+BERT (PhoBERT). "
            "Upload ảnh và đặt câu hỏi để nhận câu trả lời từ AI."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model khi khởi tạo app
    engine: Optional[VQAInferenceEngine] = None
    try:
        engine = VQAInferenceEngine(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info("VQA Engine đã khởi tạo thành công.")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo engine: {e}")

    # ======================== Endpoints ========================

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        """Kiểm tra trạng thái hệ thống và model."""
        if engine is not None:
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                model_info=engine.get_model_info(),
            )
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_info=None,
        )

    @app.get(
        "/model/info", response_model=ModelInfoResponse, tags=["Model"]
    )
    async def get_model_info() -> ModelInfoResponse:
        """Lấy thông tin chi tiết về model đang hoạt động."""
        if engine is None:
            raise HTTPException(
                status_code=503, detail="Model chưa được load."
            )
        info: Dict[str, Any] = engine.get_model_info()
        return ModelInfoResponse(**info)

    @app.post(
        "/predict", response_model=PredictionResponse, tags=["Inference"]
    )
    async def predict(
        image: UploadFile = File(..., description="File ảnh (JPEG, PNG)"),
        question: str = Form(..., description="Câu hỏi về ảnh"),
        top_k: int = Form(
            default=5, ge=1, le=20, description="Số câu trả lời top-k"
        ),
    ) -> PredictionResponse:
        """
        Dự đoán câu trả lời cho một cặp (ảnh, câu hỏi).

        - Upload ảnh dạng JPEG/PNG
        - Nhập câu hỏi bằng tiếng Anh hoặc tiếng Việt
        - Trả về top-k câu trả lời kèm confidence
        """
        if engine is None:
            raise HTTPException(
                status_code=503, detail="Model chưa được load."
            )

        # Validate file type
        allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Định dạng ảnh không hỗ trợ: {image.content_type}. "
                f"Hỗ trợ: {', '.join(allowed_types)}",
            )

        try:
            # Đọc ảnh
            image_bytes: bytes = await image.read()
            pil_image: Image.Image = Image.open(io.BytesIO(image_bytes)).convert(
                "RGB"
            )

            # Dự đoán
            result: Dict[str, Any] = engine.predict(
                image=pil_image,
                question=question.strip(),
                top_k=top_k,
            )

            return PredictionResponse(**result)

        except Exception as e:
            logger.error(f"Lỗi dự đoán: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Lỗi xử lý: {str(e)}"
            )

    @app.post(
        "/predict/url", response_model=PredictionResponse, tags=["Inference"]
    )
    async def predict_from_url(
        image_path: str = Form(
            ..., description="Đường dẫn hoặc URL ảnh trên server"
        ),
        question: str = Form(..., description="Câu hỏi về ảnh"),
        top_k: int = Form(
            default=5, ge=1, le=20, description="Số câu trả lời top-k"
        ),
    ) -> PredictionResponse:
        """
        Dự đoán câu trả lời sử dụng đường dẫn ảnh trên server.

        - Sử dụng đường dẫn file ảnh cục bộ
        - Phù hợp cho tích hợp hệ thống nội bộ
        """
        if engine is None:
            raise HTTPException(
                status_code=503, detail="Model chưa được load."
            )

        if not Path(image_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Không tìm thấy ảnh: {image_path}"
            )

        try:
            result: Dict[str, Any] = engine.predict(
                image=image_path,
                question=question.strip(),
                top_k=top_k,
            )
            return PredictionResponse(**result)

        except Exception as e:
            logger.error(f"Lỗi dự đoán: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Lỗi xử lý: {str(e)}"
            )

    return app


# ======================== Entry Point ========================

if __name__ == "__main__":
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="VQA FastAPI REST API Server"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Đường dẫn config JSON",
    )
    parser.add_argument(
        "-r",
        "--resume",
        required=True,
        type=str,
        help="Đường dẫn checkpoint .pth",
    )
    parser.add_argument(
        "--device", default=None, type=str, help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", type=str, help="Host"
    )
    parser.add_argument(
        "--port", default=8000, type=int, help="Port"
    )
    parser.add_argument(
        "--workers", default=1, type=int, help="Số workers"
    )

    args = parser.parse_args()

    app = create_app(
        config_path=args.config,
        checkpoint_path=args.resume,
        device=args.device,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )
