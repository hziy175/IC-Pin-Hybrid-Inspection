from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DESKEWED_DIR = OUTPUT_DIR / "deskewed"
CNN_DIAGNOSTIC_DIR = OUTPUT_DIR / "cnn_diagnostic"
RESULT_DIR = OUTPUT_DIR / "results"
BATCH_DIR = OUTPUT_DIR / "batch"
MODEL_PATH = MODEL_DIR / "resnet18_pin_demo.onnx"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
