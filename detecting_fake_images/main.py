from PIL import Image
from detectors.prithivml import DFDetectorV2
from detectors.proposed_detector import ProposedDetector
from detectors.susy import SuSy
from detectors.trufor import TruForDetector
from sklearn.metrics import classification_report
from time import sleep
from tqdm import tqdm
from transformers import logging
from typing import List, Tuple
from utils import crop_to_square_and_resize, find_optimal_threshold
import numpy as np
import os
import traceback
import warnings

logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Importing from timm\.models\.layers.*",
)


def get_all_image_paths(dir_path: str) -> List[str]:
    output: List[str] = []
    for p in sorted(os.listdir(dir_path)):
        if os.path.isfile(os.path.join(dir_path, p)):
            output.append(os.path.join(dir_path, p))
    return output


REAL_IMAGES_PATH = "../image_text_box/inputs"
FAKE_IMAGES_PATH = "../flux-inpaint/auto_comfyui/results"
# FAKE_IMAGES_PATH = "../flux-inpaint/auto_comfyui/results2"

LOAD_IMAGES_LIMIT = 10

real_image_paths = get_all_image_paths(REAL_IMAGES_PATH)[:LOAD_IMAGES_LIMIT]
fake_image_paths = get_all_image_paths(FAKE_IMAGES_PATH)[:LOAD_IMAGES_LIMIT]


y_true = np.array([1] * len(real_image_paths) + [0] * len(fake_image_paths))
all_paths = real_image_paths + fake_image_paths

Detectors: List[Tuple[str, object]] = [
    ("Deep-Fake-Detector-v2", DFDetectorV2()),
    ("SuSy", SuSy(top_k_patches=16, plot_worst_patches=0)),
    ("TruFor", TruForDetector()),
    ("ProposedDetector", ProposedDetector()),
]

for name, model in Detectors:
    scores = []
    for path in tqdm(all_paths, desc=f"Running {name}"):
        try:
            scores.append(model.is_real(crop_to_square_and_resize(Image.open(path))))
        except Exception as e:
            print(traceback.format_exc())
            print(path)
            y_pred.append(0)
    scores = np.array(scores)
    
    best_thr = find_optimal_threshold(y_true, scores)
    print(f"Optimal F1 threshold {name} = {best_thr:.3f}")
    
    y_pred = scores > best_thr

    report = classification_report(
        y_true,
        y_pred,
        target_names=["fake", "real"],
        digits=3,
        zero_division=0
    )
    print(f"{name} \n{report}\n" + "-" * 80)
    sleep(0.1)
