import os
import torch
from PIL import Image
from torchvision import transforms

from .trufor_lib.config import config
from .trufor_lib.utils import get_model

class TruForDetector:
    def __init__(self, model_file: str = "trufor.pth.tar", device: str = None, save_np: bool = False):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        if device != 'cpu':
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = config.CUDNN.BENCHMARK
            cudnn.deterministic = config.CUDNN.DETERMINISTIC
            cudnn.enabled = config.CUDNN.ENABLED
        self.save_np = save_np

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' not found")

        config.defrost()
        config.TEST.MODEL_FILE = model_file
        config.merge_from_file(f'detectors/trufor_lib/config/trufor_ph3.yaml')
        config.freeze()

        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        self.model = get_model(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def is_real(self, image: Image.Image) -> float:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred, conf, det, npp = self.model(img_tensor, save_np=self.save_np)

            det_sig = torch.sigmoid(det).item()
            return 1.0 - float(det_sig)

    def score_to_bool(self, score: float, threshold: float = 0.634) -> bool:
        return score > threshold
