import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F

from utils import crop_to_square_and_resize

class ProposedDetector:
    def __init__(self, weights_path="proposed_detector_all_images.pth", device=None):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Model definition must match training
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
    
        # Preprocessing should match your training transform
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: crop_to_square_and_resize(img)),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def is_real(self, image: Image.Image) -> float:
        """
        Returns a realness score in [0, 1]. Higher = more real.
        """
        img = self.transform(image).unsqueeze(0).to(self.device)  # shape: (1, C, H, W)
        with torch.no_grad():
            logits = self.model(img)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        real_prob = float(probs[0])
        # fake_prob = float(probs[1])
        return real_prob
    
    def score_to_bool(self, score: float, threshold: float = 0.604) -> bool:
        return score > threshold