"""MNIST Student Model — Gradio Web UI

启动:
    python app_gradio.py

依赖:
    pip install gradio onnxruntime pillow numpy

模型加载顺序（自动选择，按优先级）:
    1. student_model_artifacts/model.onnx        ← 推荐，无需 PyTorch
    2. student_model_artifacts/pytorch_model.bin ← fallback，需要 PyTorch
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

ARTIFACTS_DIR = Path(__file__).parent / "student_model_artifacts"
ONNX_PATH = ARTIFACTS_DIR / "model.onnx"
PT_PATH = ARTIFACTS_DIR / "pytorch_model.bin"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"


# =====================================================================
# 加载模型 —— 优先 ONNX Runtime，fallback 到 PyTorch
# =====================================================================
class Predictor:
    def __init__(self) -> None:
        if not ARTIFACTS_DIR.exists():
            raise FileNotFoundError(
                f"未找到模型目录 {ARTIFACTS_DIR}\n"
                f"请先运行 `python distilling_model.py` 训练并导出模型。"
            )

        self.labels = [str(i) for i in range(10)]
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                id2label = cfg.get("id2label") or {}
                if id2label:
                    self.labels = [id2label[str(i)] for i in range(len(id2label))]
            except Exception:
                pass

        self.backend, self.runner = self._load_backend()

    def _load_backend(self):
        if ONNX_PATH.exists():
            try:
                import onnxruntime as ort

                providers = ["CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)
                input_name = sess.get_inputs()[0].name

                def run(arr: np.ndarray) -> np.ndarray:
                    return sess.run(None, {input_name: arr})[0]

                return f"ONNX Runtime ({providers[0]})", run
            except ImportError:
                print("[warn] onnxruntime 未安装，尝试使用 PyTorch backend")

        if PT_PATH.exists():
            import torch
            import torch.nn as nn

            class StudentModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(784, 20)
                    self.fc2 = nn.Linear(20, 20)
                    self.fc3 = nn.Linear(20, 10)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    x = x.view(-1, 784)
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    return self.fc3(x)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = StudentModel().to(device)
            net.load_state_dict(torch.load(PT_PATH, map_location=device))
            net.eval()

            def run(arr: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    t = torch.from_numpy(arr).to(device)
                    return net(t).cpu().numpy()

            return f"PyTorch ({device})", run

        raise FileNotFoundError(
            f"在 {ARTIFACTS_DIR} 中既没找到 model.onnx 也没找到 pytorch_model.bin\n"
            f"请先运行 `python distilling_model.py` 训练并导出模型。"
        )

    def predict(self, arr: np.ndarray) -> np.ndarray:
        logits = self.runner(arr)
        # softmax
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


# =====================================================================
# 图像预处理 —— 复现 MNIST 标准预处理
#
# LeCun 的 MNIST 数据集对样本做过严格标准化：
#   1. 黑底白字
#   2. 数字主体在 20x20 内（保持长宽比）
#   3. 按质心居中放到 28x28 画布
#
# 任意来源图（上传/画板）都必须复现这套预处理，否则精度断崖式下降。
# =====================================================================
_BG_THRESHOLD = 30   # 像素值低于此视为背景
_FG_PADDING_BBOX = 2 # 裁剪 bbox 时向外膨胀 2 像素，避免笔画被切到边缘


def _trim_alpha_to_bbox(img: Image.Image) -> Image.Image | None:
    """如果图像带 alpha 通道（如 Sketchpad 的透明背景），用 alpha 通道找到笔画 bbox 并把背景填黑。"""
    if img.mode in ("RGBA", "LA"):
        alpha = np.asarray(img.split()[-1])
        if alpha.max() == 0:
            return None
        bg = Image.new("L", img.size, 0)
        gray = img.convert("L")
        bg.paste(gray, mask=img.split()[-1])
        return bg
    return None


def to_mnist_tensor(img: Image.Image, invert_if_needed: bool = True) -> Tuple[np.ndarray, Image.Image]:
    """把任意 PIL 图像规整为 MNIST 风格的 [1,1,28,28] float32 张量。

    Returns:
        tensor: shape [1,1,28,28]，值在 [0,1]，黑底白字
        preview: 28x28 PIL 灰度图，方便 UI 显示模型"看到"了什么
    """
    alpha_processed = _trim_alpha_to_bbox(img)
    if alpha_processed is not None:
        gray = alpha_processed
    else:
        gray = img.convert("L")
        if invert_if_needed:
            arr_full = np.asarray(gray, dtype=np.float32)
            # 启发式判断"白底黑笔" -> 反相成"黑底白笔"
            if arr_full.mean() > 127:
                gray = ImageOps.invert(gray)

    arr = np.asarray(gray, dtype=np.float32)

    ys, xs = np.where(arr > _BG_THRESHOLD)
    if len(xs) == 0:
        empty = Image.new("L", (28, 28), 0)
        return np.zeros((1, 1, 28, 28), dtype=np.float32), empty

    pad = _FG_PADDING_BBOX
    H, W = arr.shape
    y0 = max(0, ys.min() - pad)
    y1 = min(H, ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(W, xs.max() + 1 + pad)
    cropped = gray.crop((x0, y0, x1, y1))

    h, w = y1 - y0, x1 - x0
    if h >= w:
        new_h = 20
        new_w = max(1, round(20 * w / h))
    else:
        new_w = 20
        new_h = max(1, round(20 * h / w))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    arr_r = np.asarray(resized, dtype=np.float32)
    total = arr_r.sum()
    if total > 0:
        cy = (arr_r.sum(axis=1) * np.arange(new_h)).sum() / total
        cx = (arr_r.sum(axis=0) * np.arange(new_w)).sum() / total
    else:
        cy, cx = new_h / 2, new_w / 2
    paste_x = int(round(14 - cx))
    paste_y = int(round(14 - cy))
    paste_x = max(0, min(28 - new_w, paste_x))
    paste_y = max(0, min(28 - new_h, paste_y))
    canvas.paste(resized, (paste_x, paste_y))

    out = np.asarray(canvas, dtype=np.float32) / 255.0
    return out.reshape(1, 1, 28, 28), canvas


# =====================================================================
# Gradio UI
# =====================================================================
def build_app():
    import gradio as gr

    predictor = Predictor()
    print(f"[info] backend = {predictor.backend}")

    def infer_image(pil_img: Image.Image | None):
        if pil_img is None:
            return None, None, "请上传图片或在画板上手写一个数字。"
        tensor, preview = to_mnist_tensor(pil_img)
        probs = predictor.predict(tensor)[0]
        top = int(probs.argmax())
        confidences = {predictor.labels[i]: float(probs[i]) for i in range(len(probs))}
        info = f"预测: **{predictor.labels[top]}**  置信度: {probs[top]:.2%}\n后端: {predictor.backend}"
        zoomed = preview.resize((140, 140), Image.Resampling.NEAREST)
        return zoomed, confidences, info

    def infer_sketchpad(sketch: dict | None):
        """sketchpad 输入是 {'background': PIL, 'composite': PIL, 'layers': [...]}。"""
        if sketch is None:
            return None, None, "请在画板上写一个数字。"
        # gradio v4+: ImageEditor / Sketchpad 返回 dict
        img = sketch.get("composite") if isinstance(sketch, dict) else sketch
        if img is None:
            return None, None, "画板内容为空。"
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        return infer_image(img)

    with gr.Blocks(title="MNIST Student Model (Distilled)") as demo:
        gr.Markdown(
            "# MNIST 数字识别\n"
            "蒸馏得到的 student 模型（20→20 MLP，约 16 KB）。\n"
            "支持 **上传图片** 或 **画板手写**。"
        )

        with gr.Tab("上传图片"):
            with gr.Row():
                with gr.Column():
                    inp_img = gr.Image(
                        type="pil",
                        image_mode="L",
                        sources=["upload", "clipboard"],
                        label="输入图片",
                        height=280,
                    )
                    btn_img = gr.Button("识别", variant="primary")
                with gr.Column():
                    preview_img = gr.Image(label="预处理预览 (28x28，模型实际看到的图)", height=140, image_mode="L")
                    label_img = gr.Label(num_top_classes=3, label="Top-3 预测")
                    info_img = gr.Markdown()
            btn_img.click(infer_image, inp_img, [preview_img, label_img, info_img])

            with gr.Accordion("📌 输入图片要求 (点击展开)", open=False):
                gr.HTML(
                    """
                    <div style="font-size: 12px; line-height: 1.6; color: #555;">
                      <p>本模型基于 <b>MNIST 数据集</b> 训练（LeCun 1998），训练样本是"白纸黑笔写的单个数字 + 居中 + 留白"风格。</p>
                      <p><b>✅ 推荐</b><br>
                        · 白纸用黑笔/铅笔手写的单个数字（手机拍照即可）<br>
                        · 屏幕上的手写数字截图（笔画清晰、对比度高）<br>
                        · <code>samples/</code> 目录下导出的 MNIST 测试样本
                      </p>
                      <p><b>⚠️ 效果较差</b><br>
                        · 印刷体数字（如电子表显示、Logo 数字）—— 模型从未见过这种风格<br>
                        · 一张图含多个数字 —— 模型只识别单个数字<br>
                        · 数字过小、过模糊、或周围有大量干扰元素
                      </p>
                      <p><b>🔧 自动预处理</b>：灰度化 → 自动反相（白底变黑底）→ bbox 裁剪 → 保持长宽比缩放至 20×20 → 质心居中粘贴到 28×28。</p>
                    </div>
                    """
                )

        with gr.Tab("画板手写"):
            with gr.Row():
                with gr.Column():
                    inp_sketch = gr.Sketchpad(
                        label="用鼠标在此处书写一个数字（白底黑笔，会自动反相）",
                        height=320,
                        image_mode="L",
                    )
                    with gr.Row():
                        btn_sketch = gr.Button("识别", variant="primary")
                        btn_clear = gr.Button("清空画板", variant="secondary")
                with gr.Column():
                    preview_sketch = gr.Image(label="预处理预览 (28x28)", height=140, image_mode="L")
                    label_sketch = gr.Label(num_top_classes=3, label="Top-3 预测")
                    info_sketch = gr.Markdown()
            btn_sketch.click(infer_sketchpad, inp_sketch, [preview_sketch, label_sketch, info_sketch])

            def reset_sketch():
                # Gradio 的 Sketchpad/ImageEditor 给 None 不会真清画布，
                # 必须显式喂一张"空白图 dict"才能触发前端重绘。
                blank = Image.new("L", (320, 320), 255)
                blank_payload = {
                    "background": blank,
                    "layers": [],
                    "composite": blank,
                }
                return blank_payload, None, None, "画板已清空，可以重新书写。"

            btn_clear.click(
                reset_sketch,
                inputs=None,
                outputs=[inp_sketch, preview_sketch, label_sketch, info_sketch],
            )

        artifact_files = sorted(p.name for p in ARTIFACTS_DIR.iterdir() if p.is_file())
        gr.HTML(
            f"""
            <div style="font-size: 12px; color: #888; margin-top: 8px;">
              <hr style="border: none; border-top: 1px solid #eee;">
              模型工件: {' · '.join(f'<code>{n}</code>' for n in artifact_files)}<br>
              推理后端: <code>{predictor.backend}</code>
            </div>
            """
        )

    return demo


def _get_lan_ips() -> list[str]:
    """获取本机局域网 IPv4 地址列表，过滤掉 127.x.x.x。"""
    import socket
    ips: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    if not ips:
        # fallback: 通过连接外网地址反查本机出口 IP（不真的发包）
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
            s.close()
        except Exception:
            pass
    return ips


if __name__ == "__main__":
    PORT = 7860
    app = build_app()

    lan_ips = _get_lan_ips()
    print("\n" + "=" * 60)
    print(" Gradio 已启动 (局域网共享模式)")
    print("=" * 60)
    print(f"  本机访问:  http://127.0.0.1:{PORT}")
    if lan_ips:
        print(f"  局域网访问 (告诉同事这个地址):")
        for ip in lan_ips:
            print(f"    -> http://{ip}:{PORT}")
    else:
        print("  [warn] 未检测到局域网 IP，请用 `ipconfig` 自行查看 IPv4 地址")
    print("=" * 60)
    print(" 提醒:")
    print("   - 仅同一 WiFi / 公司内网的人能访问")
    print("   - Windows 防火墙首次会弹窗，请勾选 [专用网络]，不要勾 [公用网络]")
    print("   - 没有身份验证。如需密码保护，给 launch() 加 auth=('user','pwd')")
    print("=" * 60 + "\n")

    app.launch(
        server_name="0.0.0.0",   # 监听所有网卡，允许局域网访问
        server_port=PORT,
        max_file_size="10mb",    # 限制上传图片体积，避免恶意大文件
        show_error=True,
        inbrowser=True,          # 自动打开本机浏览器
    )
