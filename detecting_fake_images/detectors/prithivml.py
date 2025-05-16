import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


class DFDetectorV2:
    def __init__(self, device=None):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
        self.processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")

    def is_real(self, image: Image.Image) -> float:
        with torch.no_grad():
            outputs = self.model(**self.processor(images=image, return_tensors="pt"))
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class

    def score_to_bool(self, score: float) -> bool:
        return score > 0.5