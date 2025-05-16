from __future__ import annotations

import os
import random
from time import sleep
from typing import List, Tuple

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

from utils import crop_to_square_and_resize


class RealFakeDataset(Dataset):
    def __init__(self, real_dir: str, fake_dir: str, transform: transforms.Compose):
        self.samples: List[Tuple[str, int]] = []
        for p in sorted(os.listdir(real_dir)):
            if os.path.isfile(os.path.join(real_dir, p)):
                self.samples.append((os.path.join(real_dir, p), 0))
        for p in sorted(os.listdir(fake_dir)):
            if os.path.isfile(os.path.join(fake_dir, p)):
                self.samples.append((os.path.join(fake_dir, p), 1))
        self.transform = transform
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    preprocess = transforms.Compose([
        transforms.Lambda(lambda img: crop_to_square_and_resize(img)),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = RealFakeDataset(
        "../image_text_box/inputs",
        "../flux-inpaint/auto_comfyui/results",
        # "../image_text_scoring/good",
        preprocess,
    )
    BATCH_SIZE = 32
    train_size: int = int(0.8 * len(dataset))
    test_size: int = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs: int = 2
    for epoch in range(epochs):
        model.train()
        train_loss: float = 0.0
        preds: List[int] = []
        targets: List[int] = []
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - train"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
        train_acc: float = accuracy_score(targets, preds)
        print(f"Epoch {epoch + 1}: train_loss={train_loss / len(train_loader.dataset):.4f} train_acc={train_acc:.4f}") # noqa
        sleep(0.1)
    model.eval()
    preds: List[int] = []
    targets: List[int] = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Test"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    print(classification_report(targets, preds, target_names=["Real", "Fake"], digits=3))
    torch.save(model.state_dict(), "proposed_detector.pth")

if __name__ == "__main__":
    main()
