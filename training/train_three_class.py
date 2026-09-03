from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from training.recovered_model import resnet34


CLASSES = ("black_skirt", "gray_coat", "white_skirt")
SEED = 20260902


class FashionDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def split_samples(data_dir: Path) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(SEED)
    train: list[tuple[Path, int]] = []
    validation: list[tuple[Path, int]] = []
    for label, category in enumerate(CLASSES):
        files = sorted((data_dir / category).glob("*.jpg"))
        if len(files) < 10:
            raise ValueError(f"Not enough training images for {category}")
        rng.shuffle(files)
        validation_count = max(1, round(len(files) * 0.2))
        validation.extend((path, label) for path in files[:validation_count])
        train.extend((path, label) for path in files[validation_count:])
    rng.shuffle(train)
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--recovered-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_samples, validation_samples = split_samples(args.data)
    print(f"train={len(train_samples)} validation={len(validation_samples)} classes={CLASSES}")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_loader = DataLoader(FashionDataset(train_samples, train_transform), batch_size=16, shuffle=True, num_workers=0)
    validation_loader = DataLoader(FashionDataset(validation_samples, validation_transform), batch_size=24, shuffle=False, num_workers=0)

    model = resnet34(num_classes=len(CLASSES))
    recovered = torch.load(args.recovered_checkpoint, map_location="cpu", weights_only=True)
    if recovered["fc.weight"].shape[0] != 5:
        raise ValueError("Recovered checkpoint does not have the documented five-output head")
    backbone = {key: value for key, value in recovered.items() if not key.startswith("fc.")}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise ValueError(f"Unexpected backbone mismatch: missing={missing}, unexpected={unexpected}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    model.to(device)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.04)
    best_accuracy = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
        with torch.inference_mode():
            for images, labels in validation_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += labels.size(0)
                for truth, prediction in zip(labels, predictions):
                    confusion[truth, prediction] += 1
        accuracy = correct / total
        print(f"epoch={epoch} loss={running_loss / len(train_samples):.4f} val_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_confusion = confusion.clone()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": "resnet34",
        "classes": CLASSES,
        "state_dict": best_state,
        "validation_accuracy": best_accuracy,
        "validation_confusion": best_confusion,
        "seed": SEED,
    }, args.output)
    print(f"saved={args.output} best_val_accuracy={best_accuracy:.4f}")
    print(best_confusion)


if __name__ == "__main__":
    main()
