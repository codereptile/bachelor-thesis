import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve


def crop_to_square_and_resize(image: Image.Image) -> Image.Image:
    np_image: np.ndarray = np.array(image)
    h: int = np_image.shape[0]
    w: int = np_image.shape[1]
    side: int = min(h, w)
    y0: int = (h - side) // 2
    x0: int = (w - side) // 2
    cropped: np.ndarray = np_image[y0: y0 + side, x0: x0 + side]
    cropped_image: Image.Image = Image.fromarray(cropped)
    resized_image: Image.Image = cropped_image.resize((1024, 1024), Image.Resampling.LANCZOS)
    return resized_image

def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    return pr_thresholds[best_idx]