from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def load_east_model(model_path: str) -> cv2.dnn.Net:
    return cv2.dnn.readNet(model_path)


def score_window(
        image_np: np.ndarray,
        box: Tuple[int, int, int, int],
        net: cv2.dnn.Net,
        input_size: Tuple[int, int] = (320, 320),
) -> float:
    x, y, w, h = box
    patch: np.ndarray = image_np[y : y + h, x : x + w]
    blob: np.ndarray = cv2.dnn.blobFromImage(
        patch,
        scalefactor=1.0,
        size=input_size,
        mean=(123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    scores: np.ndarray = net.forward("feature_fusion/Conv_7/Sigmoid")
    score_map: np.ndarray = scores[0, 0]
    score_map_resized: np.ndarray = cv2.resize(
        score_map, (w, h), interpolation=cv2.INTER_LINEAR
    )
    return float(np.mean(score_map_resized))


def crop_to_square(image: np.ndarray) -> np.ndarray:
    h: int = image.shape[0]
    w: int = image.shape[1]
    side: int = min(h, w)
    y0: int = (h - side) // 2
    x0: int = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def generate_windows(
        image_shape: Tuple[int, int],
        window_sizes: List[Tuple[int, int]],
        step: int,
) -> List[Tuple[int, int, int, int]]:
    h, w = image_shape
    boxes: List[Tuple[int, int, int, int]] = []
    for win_w, win_h in window_sizes:
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                boxes.append((x, y, win_w, win_h))
    return boxes


def mask_outside_box(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    masked: np.ndarray = np.zeros_like(image)
    masked[y : y + h, x : x + w] = image[y : y + h, x : x + w]
    return masked


def transparent_inside_box(
        image: np.ndarray, box: Tuple[int, int, int, int]
) -> np.ndarray:
    x, y, w, h = box
    rgba: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[y : y + h, x : x + w, 3] = 0
    return rgba


def prepare_output_dir(dir_path: str) -> None:
    path: Path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    INPUT_DIR = "inputs"
    OUTPUT_DIR = "../flux-inpaint/auto_comfyui/input"
    
    net_east_model: cv2.dnn.Net = load_east_model("frozen_east_text_detection.pb")
    image_dir: Path = Path(INPUT_DIR)
    image_paths: List[Path] = [p for p in sorted(image_dir.iterdir()) if p.is_file()]
    prepare_output_dir(OUTPUT_DIR)
    window_sizes: List[Tuple[int, int]] = [(800, 800), (640, 640), (800, 480), (800, 320)]
    step: int = 32
    
    OUTPUT_BEST_COUNT = 1

    for img_path in tqdm(image_paths, desc="Processing all images"):
        stem: str = img_path.stem
        output_dir: Path = Path(OUTPUT_DIR)
        exists_all: bool = all(
            output_dir.joinpath(f"{stem}_transparent_{i+1}.png").exists() for i in range(OUTPUT_BEST_COUNT)
        )
        if exists_all:
            continue

        img: np.ndarray = cv2.imread(str(img_path))
        if img is None:
            continue

        img_sq: np.ndarray = crop_to_square(img)
        img_resized: np.ndarray = cv2.resize(
            img_sq, (1024, 1024), interpolation=cv2.INTER_AREA
        )
        boxes: List[Tuple[int, int, int, int]] = generate_windows(
            img_resized.shape[:2], window_sizes, step
        )
        scores: List[float] = []
        for box in boxes:
            score: float = score_window(img_resized, box, net_east_model)
            scores.append(score)

        sorted_indices: np.ndarray = np.argsort(scores)[::-1]
        top3_indices: np.ndarray = sorted_indices[:3]
        top_boxes: List[Tuple[int, int, int, int]] = [boxes[i] for i in top3_indices]
        top_scores: List[float] = [scores[i] for i in top3_indices]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes_flat = axes.flatten()
        axes_flat[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        axes_flat[0].set_title("Original")
        for idx in range(3):
            masked: np.ndarray = mask_outside_box(img_resized, top_boxes[idx])
            axes_flat[idx + 1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
            axes_flat[idx + 1].set_title(f"Rank {idx+1}\nScore {top_scores[idx]:.4f}")
        for ax in axes_flat:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        for idx in range(OUTPUT_BEST_COUNT):
            transparent: np.ndarray = transparent_inside_box(img_resized, top_boxes[idx])
            out_path: Path = output_dir.joinpath(f"{stem}_transparent_{idx+1}.png")
            cv2.imwrite(str(out_path), transparent)


if __name__ == "__main__":
    main()
