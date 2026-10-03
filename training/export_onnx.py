from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from dataset import image_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "resnet18_pin_classifier.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "resnet18_pin_demo.onnx"
TEST_IMAGE = PROJECT_ROOT / "data" / "pin_cls_raw" / "normal" / "T08_reference.jpg"


def build_model(checkpoint):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    return image_transform(image).unsqueeze(0)


def softmax_numpy(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    input_tensor = load_tensor(TEST_IMAGE)

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        input_tensor,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        torch_logits = model(input_tensor).numpy()

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: input_tensor.numpy().astype(np.float32)})[0]

    torch_prob = softmax_numpy(torch_logits)
    onnx_prob = softmax_numpy(onnx_logits)
    max_diff = float(np.max(np.abs(torch_prob - onnx_prob)))

    print(f"ONNX saved to: {ONNX_PATH}")
    print(f"Maximum probability difference: {max_diff:.8f}")


if __name__ == "__main__":
    main()
