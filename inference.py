"""
Module suy luận (Inference) cho Visual Question Answering.
Cung cấp pipeline hoàn chỉnh: load model, tiền xử lý, dự đoán.

Hỗ trợ 3 kiến trúc: MLP, CNN+LSTM, ViT+BERT.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import model as module_arch
from data_loader.data_loaders import build_answer_vocab
from utils import read_json

logger = logging.getLogger(__name__)


class VQAInferenceEngine:
    """
    Engine suy luận cho Visual Question Answering.

    Quản lý toàn bộ pipeline từ load model, tiền xử lý ảnh/câu hỏi,
    đến sinh câu trả lời với confidence score.

    Attributes:
        model: Mô hình VQA đã load.
        device: Thiết bị tính toán (CPU/GPU).
        idx_to_ans: Dict ánh xạ index -> câu trả lời.
        transform: Pipeline tiền xử lý ảnh.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> None:
        """
        Khởi tạo engine suy luận.

        Args:
            config_path: Đường dẫn tới file config JSON.
            checkpoint_path: Đường dẫn tới file checkpoint (.pth).
            device: Thiết bị ('cuda', 'cpu', hoặc None để tự phát hiện).
        """
        self.config: Dict[str, Any] = read_json(Path(config_path))
        self.checkpoint_path: str = checkpoint_path

        # Thiết lập device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(f"Sử dụng thiết bị: {self.device}")

        # Xây dựng vocab và transform
        self._build_vocab()
        self._build_transform()

        # Load model
        self._load_model()

    def _build_vocab(self) -> None:
        """Xây dựng bộ từ vựng câu trả lời từ dữ liệu training/validation."""
        data_dir: str = self.config["data_loader"]["args"]["data_dir"]
        train_json: str = str(Path(data_dir) / "train.json")
        val_json: str = str(Path(data_dir) / "val.json")

        self.vocab, self.idx_to_ans = build_answer_vocab(train_json, val_json)
        self.num_classes: int = len(self.vocab)
        logger.info(f"Số lượng lớp câu trả lời: {self.num_classes}")

    def _build_transform(self) -> None:
        """Xây dựng pipeline tiền xử lý ảnh cho inference."""
        img_size: int = self.config["data_loader"]["args"].get("img_size", 224)
        self.img_size: int = img_size
        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_model(self) -> None:
        """Load mô hình từ checkpoint và chuyển sang chế độ eval."""
        model_type: str = self.config["arch"]["type"]
        model_args: Dict[str, Any] = dict(self.config["arch"]["args"])

        # Chuẩn bị train_questions nếu model cần (MLP, CNN+LSTM)
        train_questions: Optional[List[str]] = None
        if model_type in ("MLPBaseline", "CNNLSTMBaseline"):
            data_dir = self.config["data_loader"]["args"]["data_dir"]
            train_json_path = str(Path(data_dir) / "train.json")
            with open(train_json_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            train_questions = [item["question"] for item in train_data]

        # Khởi tạo model theo kiến trúc
        if model_type == "MLPBaseline":
            self.model = module_arch.MLPBaseline(
                self.num_classes,
                train_questions=train_questions,
                **model_args,
            )
        elif model_type == "CNNLSTMBaseline":
            self.model = module_arch.CNNLSTMBaseline(
                self.num_classes,
                train_questions=train_questions,
                **model_args,
            )
        elif model_type == "ViTBERTBaseline":
            self.model = module_arch.ViTBERTBaseline(
                self.num_classes, **model_args
            )
        else:
            raise ValueError(f"Kiến trúc không được hỗ trợ: {model_type}")

        # Load state dict từ checkpoint
        logger.info(f"Đang load checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"]

        # Xử lý trường hợp model được lưu với DataParallel
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Model {model_type} đã load thành công - "
            f"Epoch: {checkpoint.get('epoch', 'N/A')}"
        )

    def preprocess_image(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Tiền xử lý ảnh đầu vào.

        Args:
            image: Ảnh đầu vào, hỗ trợ nhiều định dạng:
                - str/Path: đường dẫn file ảnh
                - PIL.Image: ảnh PIL
                - np.ndarray: mảng numpy (H, W, C) dạng RGB

        Returns:
            Tensor ảnh đã tiền xử lý, shape (1, 3, img_size, img_size).
        """
        if isinstance(image, (str, Path)):
            pil_image: Image.Image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(
                f"Định dạng ảnh không được hỗ trợ: {type(image)}"
            )

        img_tensor: torch.Tensor = self.transform(pil_image).unsqueeze(0)
        return img_tensor.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Dự đoán câu trả lời cho một cặp (ảnh, câu hỏi).

        Args:
            image: Ảnh đầu vào (đường dẫn, PIL Image, hoặc numpy array).
            question: Câu hỏi về ảnh.
            top_k: Số lượng câu trả lời top-k cần trả về.

        Returns:
            Dict chứa:
                - answer: Câu trả lời tốt nhất (str).
                - confidence: Độ tin cậy của câu trả lời (float).
                - top_k_answers: List các câu trả lời top-k kèm confidence.
                - inference_time_ms: Thời gian suy luận (ms).
        """
        start_time: float = time.time()

        # Tiền xử lý
        img_tensor: torch.Tensor = self.preprocess_image(image)
        questions: List[str] = [question]

        # Suy luận
        output: torch.Tensor = self.model(img_tensor, questions)
        probabilities: torch.Tensor = torch.softmax(output, dim=1)

        # Lấy top-k kết quả
        k: int = min(top_k, probabilities.size(1))
        top_k_probs, top_k_indices = probabilities.topk(k, dim=1)

        top_k_answers: List[Dict[str, Any]] = []
        for i in range(k):
            idx: int = top_k_indices[0, i].item()
            prob: float = top_k_probs[0, i].item()
            answer_text: str = self.idx_to_ans.get(idx, "<UNKNOWN>")
            top_k_answers.append(
                {
                    "answer": answer_text,
                    "confidence": round(prob * 100, 2),
                    "rank": i + 1,
                }
            )

        inference_time_ms: float = (time.time() - start_time) * 1000

        return {
            "answer": top_k_answers[0]["answer"],
            "confidence": top_k_answers[0]["confidence"],
            "top_k_answers": top_k_answers,
            "inference_time_ms": round(inference_time_ms, 2),
            "model_type": self.config["arch"]["type"],
            "question": question,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        questions: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Dự đoán câu trả lời cho một batch (ảnh, câu hỏi).

        Args:
            images: Danh sách ảnh đầu vào.
            questions: Danh sách câu hỏi tương ứng.
            top_k: Số lượng câu trả lời top-k cần trả về cho mỗi cặp.

        Returns:
            Danh sách Dict kết quả cho mỗi cặp (ảnh, câu hỏi).
        """
        if len(images) != len(questions):
            raise ValueError(
                f"Số lượng ảnh ({len(images)}) và câu hỏi ({len(questions)}) "
                "phải bằng nhau."
            )

        start_time: float = time.time()

        # Tiền xử lý batch ảnh
        img_tensors: List[torch.Tensor] = [
            self.preprocess_image(img).squeeze(0) for img in images
        ]
        img_batch: torch.Tensor = torch.stack(img_tensors).to(self.device)

        # Suy luận batch
        output: torch.Tensor = self.model(img_batch, questions)
        probabilities: torch.Tensor = torch.softmax(output, dim=1)

        # Xử lý kết quả
        batch_size: int = len(images)
        k: int = min(top_k, probabilities.size(1))
        top_k_probs, top_k_indices = probabilities.topk(k, dim=1)

        results: List[Dict[str, Any]] = []
        for b in range(batch_size):
            top_k_answers: List[Dict[str, Any]] = []
            for i in range(k):
                idx: int = top_k_indices[b, i].item()
                prob: float = top_k_probs[b, i].item()
                answer_text: str = self.idx_to_ans.get(idx, "<UNKNOWN>")
                top_k_answers.append(
                    {
                        "answer": answer_text,
                        "confidence": round(prob * 100, 2),
                        "rank": i + 1,
                    }
                )

            results.append(
                {
                    "answer": top_k_answers[0]["answer"],
                    "confidence": top_k_answers[0]["confidence"],
                    "top_k_answers": top_k_answers,
                    "question": questions[b],
                }
            )

        total_time_ms: float = (time.time() - start_time) * 1000
        avg_time_ms: float = total_time_ms / batch_size

        for r in results:
            r["inference_time_ms"] = round(avg_time_ms, 2)
            r["model_type"] = self.config["arch"]["type"]

        logger.info(
            f"Batch inference: {batch_size} mẫu - "
            f"Tổng: {total_time_ms:.1f}ms - "
            f"TB: {avg_time_ms:.1f}ms/mẫu"
        )

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Trả về thông tin chi tiết về model đang sử dụng.

        Returns:
            Dict chứa thông tin model: kiến trúc, tham số, device, v.v.
        """
        trainable_params: int = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params: int = sum(p.numel() for p in self.model.parameters())

        return {
            "model_type": self.config["arch"]["type"],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_answer_classes": self.num_classes,
            "image_size": self.img_size,
            "device": str(self.device),
            "checkpoint": self.checkpoint_path,
        }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="VQA Inference CLI")
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Đường dẫn config JSON"
    )
    parser.add_argument(
        "-r", "--resume", required=True, type=str, help="Đường dẫn checkpoint"
    )
    parser.add_argument(
        "-i", "--image", required=True, type=str, help="Đường dẫn ảnh"
    )
    parser.add_argument(
        "-q", "--question", required=True, type=str, help="Câu hỏi về ảnh"
    )
    parser.add_argument(
        "--top_k", default=5, type=int, help="Số câu trả lời top-k"
    )
    parser.add_argument(
        "--device", default=None, type=str, help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    engine = VQAInferenceEngine(
        config_path=args.config,
        checkpoint_path=args.resume,
        device=args.device,
    )

    result = engine.predict(args.image, args.question, top_k=args.top_k)

    print("\n" + "=" * 60)
    print(f"  Câu hỏi: {result['question']}")
    print(f"  Câu trả lời: {result['answer']}")
    print(f"  Độ tin cậy: {result['confidence']}%")
    print(f"  Thời gian: {result['inference_time_ms']}ms")
    print(f"  Model: {result['model_type']}")
    print("=" * 60)
    print("\nTop-K câu trả lời:")
    for ans in result["top_k_answers"]:
        print(f"  #{ans['rank']}: {ans['answer']} ({ans['confidence']}%)")
