from PIL import Image
from detectors.prithivml import DFDetectorV2
from detectors.proposed_detector import ProposedDetector
from detectors.susy import SuSy
from detectors.trufor import TruForDetector
from transformers import logging
from typing import List, Tuple
from utils import crop_to_square_and_resize


logging.set_verbosity_error()

Detectors: List[Tuple[str, object]] = [
    ("SuSy", SuSy(top_k_patches=16, plot_worst_patches=0)),
    ("Deep-Fake-Detector-v2", DFDetectorV2()),
    ("ProposedDetector", ProposedDetector()),
    ("TruFor", TruForDetector()),
]

while True:
    path = input("Enter path:")
    for name, model in Detectors:
        score = model.is_real(crop_to_square_and_resize(Image.open(path)))
        is_real = model.score_to_bool(score)
        print(f"{name:<22} thinks it's {"REAL" if is_real else "FAKE"} (score: {score:.3f})")
