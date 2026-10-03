from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "pin_cls_raw"
NORMAL_DIR = DATASET_ROOT / "normal"
DEFECT_DIR = DATASET_ROOT / "defect_synthetic"


class PadToSquareByEdge:
    def __call__(self, image):
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        if height == width:
            return image

        if height > width:
            difference = height - width
            left = difference // 2
            right = difference - left
            top = bottom = 0
        else:
            difference = width - height
            top = difference // 2
            bottom = difference - top
            left = right = 0

        padded = cv2.copyMakeBorder(
            image_array,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(padded)


image_transform = transforms.Compose(
    [
        PadToSquareByEdge(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class PinDataset(Dataset):
    def __init__(self, normal_dir=NORMAL_DIR, defect_dir=DEFECT_DIR, transform=image_transform):
        self.transform = transform
        self.samples = []

        for image_path in sorted(Path(normal_dir).glob("*.jpg")):
            self.samples.append({"path": image_path, "label": 0, "label_name": "normal"})

        for image_path in sorted(Path(defect_dir).glob("*.jpg")):
            self.samples.append({"path": image_path, "label": 1, "label_name": "defect"})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample["label"], str(sample["path"])
