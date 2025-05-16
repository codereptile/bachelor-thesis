from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision import models, transforms
from tqdm.auto import tqdm


def crop_to_square_np(img: np.ndarray) -> np.ndarray:
    h: int = img.shape[0]
    w: int = img.shape[1]
    side: int = min(h, w)
    y0: int = (h - side) // 2
    x0: int = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def crop_to_square(img: Image.Image) -> Image.Image:
    arr: np.ndarray = np.array(img)
    arr_sq: np.ndarray = crop_to_square_np(arr)
    return Image.fromarray(arr_sq)


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    model: torch.nn.Module = models.resnet50(weights=None)
    num_features: int = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def preprocess_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict(model: torch.nn.Module, files: List[str], device: torch.device) -> None:
    transform: transforms.Compose = preprocess_transform()
    label_names: List[str] = ["fake", "real"]
    for fp in files:
        img: Image.Image = Image.open(fp).convert("RGB")
        img = crop_to_square(img)
        img_t: Tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out: Tensor = model(img_t)
            pred: int = torch.argmax(out, dim=1).item()
        print(f"{fp} -> {label_names[pred]}")


def main() -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files: List[str] = [
        "flux_detection_reference_image_01.jpg",
        "flux_detection_reference_image_02.jpg",
        "flux_detection_reference_image_03.jpg",
        "flux_detection_reference_image_04.jpg",
        "flux_detection_reference_image_05.jpg",
        "stars_billboard_best_output.png",
        "stars_billboard_source_image.jpg",
        "../flux-inpaint/billboard_stars_output_01.png",
        "../flux-inpaint/billboard_stars_output_02.png",
        "../flux-inpaint/billboard_stars_output_03.png",
        "../image_text_box/shops/shop_0001.jpg",
        "../image_text_box/shops/shop_0002.jpg",
        "../image_text_box/shops/shop_0003.jpg",
        "../flux-inpaint/auto_comfyui/results/shop_0001_transparent_1_seed1_steps20.png",
        "../flux-inpaint/auto_comfyui/results/shop_0002_transparent_1_seed1_steps20.png",
        "../flux-inpaint/auto_comfyui/results/shop_0003_transparent_1_seed1_steps20.png",
    ]
    model: torch.nn.Module = load_model("resnet50_real_vs_fake.pt", device)
    predict(model, files, device)


if __name__ == "__main__":
    main()
