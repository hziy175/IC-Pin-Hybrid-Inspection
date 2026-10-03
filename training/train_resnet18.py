from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import PinDataset, image_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "resnet18_pin_classifier.pth"

BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_random_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PinDataset(transform=image_transform)
    normal_count = sum(sample["label"] == 0 for sample in dataset.samples)
    defect_count = sum(sample["label"] == 1 for sample in dataset.samples)

    if normal_count == 0 or defect_count == 0:
        raise RuntimeError("Both normal and defect samples are required.")

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    total_count = normal_count + defect_count
    class_weights = torch.tensor(
        [
            total_count / (2.0 * normal_count),
            total_count / (2.0 * defect_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}")
    print(f"Samples: normal={normal_count}, defect={defect_count}, total={len(dataset)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_count = 0
        sample_count = 0

        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size
            correct_count += (torch.argmax(logits, dim=1) == labels).sum().item()

        epoch_loss = running_loss / sample_count
        epoch_accuracy = correct_count / sample_count
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={epoch_loss:.4f} | training_accuracy={epoch_accuracy * 100:.1f}%"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["normal", "defect"],
            "normal_label": 0,
            "defect_label": 1,
            "patch_width": 20,
            "patch_height": 40,
            "dataset_scope": "single-reference IC with synthetic defects; no independent validation set",
        },
        CHECKPOINT_PATH,
    )
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
