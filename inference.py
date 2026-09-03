from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


CLASSES = ("black_skirt", "gray_coat", "white_skirt")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class FashionClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Expected a versioned FashionVote checkpoint")
        if tuple(checkpoint.get("classes", ())) != CLASSES:
            raise ValueError("Checkpoint classes do not match the application classes")

        state_dict = checkpoint["state_dict"]
        if state_dict["fc.weight"].shape != (len(CLASSES), model.fc.in_features):
            raise ValueError("Checkpoint classifier head is not three-class")

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def predict(self, image: Image.Image) -> Prediction:
        tensor = self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor)[0], dim=0)

        values = {
            label: float(probabilities[index].item())
            for index, label in enumerate(CLASSES)
        }
        label = max(values, key=values.get)
        return Prediction(label=label, confidence=values[label], probabilities=values)
