"""
Gradio Web Application cho Visual Question Answering.
Giao diện tương tác cho phép người dùng upload ảnh, đặt câu hỏi
và nhận câu trả lời từ mô hình VQA đã huấn luyện.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from inference import VQAInferenceEngine

logger = logging.getLogger(__name__)

# Biến toàn cục để lưu engine
engine: Optional[VQAInferenceEngine] = None


def load_model(
    config_path: str, checkpoint_path: str, device: Optional[str] = None
) -> str:
    """
    Load model VQA từ config và checkpoint.

    Args:
        config_path: Đường dẫn file config JSON.
        checkpoint_path: Đường dẫn file checkpoint .pth.
        device: Thiết bị tính toán.

    Returns:
        Thông báo trạng thái load model.
    """
    global engine
    try:
        engine = VQAInferenceEngine(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        info: Dict[str, Any] = engine.get_model_info()
        return (
            f"✅ Model loaded thành công!\n"
            f"  Kiến trúc: {info['model_type']}\n"
            f"  Tham số huấn luyện: {info['trainable_parameters']:,}\n"
            f"  Số lớp trả lời: {info['num_answer_classes']}\n"
            f"  Thiết bị: {info['device']}"
        )
    except Exception as e:
        return f"❌ Lỗi load model: {str(e)}"


def predict_answer(
    image: Any, question: str, top_k: int = 5
) -> Tuple[str, str, str]:
    """
    Dự đoán câu trả lời cho ảnh và câu hỏi.

    Args:
        image: Ảnh đầu vào từ Gradio.
        question: Câu hỏi về ảnh.
        top_k: Số câu trả lời top-k.

    Returns:
        Tuple (answer, details, top_k_table) để hiển thị trên Gradio.
    """
    if engine is None:
        return "⚠️ Model chưa được load!", "", ""

    if image is None:
        return "⚠️ Vui lòng upload một ảnh!", "", ""

    if not question or not question.strip():
        return "⚠️ Vui lòng nhập câu hỏi!", "", ""

    try:
        result: Dict[str, Any] = engine.predict(
            image=image,
            question=question.strip(),
            top_k=top_k,
        )

        # Câu trả lời chính
        main_answer: str = (
            f"🎯 {result['answer']}\n"
            f"📊 Độ tin cậy: {result['confidence']}%"
        )

        # Chi tiết inference
        details: str = (
            f"⏱️ Thời gian suy luận: {result['inference_time_ms']}ms\n"
            f"🤖 Model: {result['model_type']}\n"
            f"❓ Câu hỏi: {result['question']}"
        )

        # Bảng top-k
        top_k_text_lines: List[str] = ["Rank | Câu trả lời | Confidence"]
        top_k_text_lines.append("-----|-------------|----------")
        for ans in result["top_k_answers"]:
            top_k_text_lines.append(
                f"#{ans['rank']} | {ans['answer']} | {ans['confidence']}%"
            )
        top_k_text: str = "\n".join(top_k_text_lines)

        return main_answer, details, top_k_text

    except Exception as e:
        logger.error(f"Lỗi dự đoán: {e}", exc_info=True)
        return f"❌ Lỗi: {str(e)}", "", ""


def build_demo(
    config_path: str, checkpoint_path: str, device: Optional[str] = None
) -> gr.Blocks:
    """
    Xây dựng Gradio Blocks demo.

    Args:
        config_path: Đường dẫn config JSON.
        checkpoint_path: Đường dẫn checkpoint .pth.
        device: Thiết bị tính toán.

    Returns:
        Gradio Blocks app.
    """
    # Load model khi khởi tạo
    status_msg: str = load_model(config_path, checkpoint_path, device)

    with gr.Blocks(
        title="VQA - Visual Question Answering",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .result-box { font-size: 18px; font-weight: bold; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🖼️ Visual Question Answering (VQA)
            ### Hệ thống trả lời câu hỏi về ảnh sử dụng Deep Learning
            
            Upload một ảnh, đặt câu hỏi và nhận câu trả lời từ AI.
            Hỗ trợ 3 kiến trúc: **MLP**, **CNN+LSTM** (ResNet50), **ViT+BERT** (PhoBERT).
            """,
            elem_classes="main-header",
        )

        # Hiển thị trạng thái model
        gr.Textbox(
            value=status_msg, label="Trạng thái Model", interactive=False
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                image_input = gr.Image(
                    label="📸 Upload Ảnh",
                    type="pil",
                    height=400,
                )
                question_input = gr.Textbox(
                    label="❓ Câu hỏi",
                    placeholder="Nhập câu hỏi về ảnh (VD: What color is the car?)",
                    lines=2,
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="🔢 Số câu trả lời Top-K",
                )
                predict_btn = gr.Button(
                    "🚀 Dự đoán", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                # Output
                answer_output = gr.Textbox(
                    label="🎯 Câu trả lời",
                    lines=3,
                    elem_classes="result-box",
                )
                details_output = gr.Textbox(
                    label="📋 Chi tiết", lines=4
                )
                topk_output = gr.Textbox(
                    label="📊 Top-K Câu trả lời", lines=8
                )

        # Ví dụ sử dụng
        gr.Markdown(
            """
            ### 💡 Hướng dẫn sử dụng
            1. **Upload ảnh** hoặc kéo thả vào khung bên trái
            2. **Nhập câu hỏi** bằng tiếng Anh hoặc tiếng Việt
            3. **Nhấn Dự đoán** để nhận câu trả lời từ AI
            4. Điều chỉnh **Top-K** để xem nhiều câu trả lời hơn
            
            ### 🏗️ Kiến trúc hỗ trợ
            | Model | Image Encoder | Text Encoder | Đặc điểm |
            |-------|--------------|-------------|-----------|
            | MLP | Flattened Pixels | TF-IDF | Baseline đơn giản |
            | CNN+LSTM | ResNet50 | Bi-LSTM | Deep learning truyền thống |
            | ViT+BERT | Vision Transformer | PhoBERT | State-of-the-art, hỗ trợ tiếng Việt |
            """
        )

        # Xử lý sự kiện
        predict_btn.click(
            fn=predict_answer,
            inputs=[image_input, question_input, top_k_slider],
            outputs=[answer_output, details_output, topk_output],
        )

        # Cho phép nhấn Enter để dự đoán
        question_input.submit(
            fn=predict_answer,
            inputs=[image_input, question_input, top_k_slider],
            outputs=[answer_output, details_output, topk_output],
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="VQA Gradio Web Application"
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
        "--device",
        default=None,
        type=str,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--port",
        default=7860,
        type=int,
        help="Port cho web server",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Tạo public link (Gradio share)",
    )

    args = parser.parse_args()

    demo = build_demo(
        config_path=args.config,
        checkpoint_path=args.resume,
        device=args.device,
    )

    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
