import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import difflib
from tqdm import tqdm


def get_mask_rect(mask_path: str, alpha_threshold: int = 128) -> Tuple[int, int, int, int]:
    """Return bounding box (x, y, w, h) of the *transparent* rectangle in the mask.

    The mask is assumed to be either:
    * RGBA with alpha = 0 for transparent pixels, OR
    * RGB where transparent area is pure white (255,255,255) – a reasonable fallback.
    """
    mask = cv2.imread(str(Path(mask_path)), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask failed to load: {mask_path}")

    # Extract transparency map
    if mask.shape[2] == 4:  # RGBA
        alpha = mask[:, :, 3]
    else:  # Fallback – assume white pixels are “transparent”
        alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        alpha = 255 - alpha  # white → 0 (transparent), dark → 255

    # Binary mask of transparent area
    transparent = (alpha < alpha_threshold).astype(np.uint8) * 255

    # Find contours and return the largest rectangle
    contours, _ = cv2.findContours(transparent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No transparent region detected in mask: {mask_path}")

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h


def upscale(img: np.ndarray, scale: float = 3.5) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def threshold_gray(gray: np.ndarray) -> np.ndarray:
    blk = 200 | 1
    C = 8
    bin_ = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, C)
    if np.mean(bin_) > 127:
        bin_ = cv2.bitwise_not(bin_)
    return bin_


def morphology(bin_: np.ndarray) -> np.ndarray:
    kernel_size = 8
    k = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed


def preprocess(img: np.ndarray) -> np.ndarray:
    img = upscale(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bin_ = threshold_gray(gray)
    proc = morphology(bin_)
    return proc


def ocr_image(img: np.ndarray) -> str:
    data = pytesseract.image_to_data(Image.fromarray(img), config="--psm 6 -l eng", output_type=pytesseract.Output.DICT)
    texts: List[str] = []
    for i, text in enumerate(data['text']):
        conf = int(data['conf'][i])
        if conf >= 50 and text.strip() != "":
            texts.append(text)
    return " ".join(texts)


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def show_pair(original: np.ndarray, crop_box: Tuple[int, int, int, int], rotated: np.ndarray, processed: np.ndarray, name: str, rotation_angle: int) -> None:
    """Visual helper: show original with ROI, rotated crop, and processed bin."""
    x, y, w, h = crop_box
    org_vis = original.copy()
    cv2.rectangle(org_vis, (x, y), (x + w, y + h), (255, 0, 0), 3)

    plt.figure(figsize=(8, 9))

    plt.subplot(3, 1, 1)
    plt.imshow(cv2.cvtColor(org_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"{name} original + ROI")
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    cw_text = 'CCW' if rotation_angle > 0 else 'CW'
    plt.title(f"crop rotated ({abs(rotation_angle)}° {cw_text})")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.imshow(processed, cmap="gray")
    plt.title("processed bin")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def check_text(text: str) -> float:
    CORRECT_TEXT = "Hello!"

    def fmt(t: str) -> str:
        return t.replace('\n', ' ').lower().strip()

    return difflib.SequenceMatcher(None, fmt(text), fmt(CORRECT_TEXT)).ratio()


def process_angle(angle: int, crop: np.ndarray) -> Tuple[float, int, np.ndarray, np.ndarray, str]:
    """Rotate → preprocess → OCR → score for a single angle, on a *cropped* ROI."""
    rot = rotate_image(crop, angle)
    proc = preprocess(rot)
    txt = ocr_image(proc)
    score = check_text(txt)
    return score, angle, rot, proc, txt


def discover_samples(results_dir: Path, inputs_dir: Path) -> List[Tuple[str, str]]:
    """Find matching (image, mask) pairs based on naming pattern.

    Accepts any result file matching:
        <stem>_transparent_<digits>_seed<digits>_steps<digits>.png
    and pairs it with inputs/<stem>_transparent_1.png.
    """
    samples: List[Tuple[str, str]] = []

    # Flexible pattern; captures the leading part before _transparent_1
    pattern_re = re.compile(
        r"^(?P<stem>.+?)_transparent_\d+_seed\d+_steps\d+\.png$",
        re.IGNORECASE,
    )

    # Use a broad glob so we don't miss files with other seed/steps numbers
    for img_path in results_dir.glob("*_transparent_1_seed*_steps*.png"):
        m = pattern_re.match(img_path.name)
        if m is None:
            print(f"[DEBUG] Filename did not match pattern: {img_path.name}")
            continue

        stem = m.group("stem")
        mask_path = inputs_dir / f"{stem}_transparent_1.png"
        if mask_path.exists():
            samples.append((str(img_path), str(mask_path)))
        else:
            print(f"[WARNING] Mask missing for {img_path.name}")

    if not samples:
        print("[INFO] No matching image/mask pairs found with current pattern.")
    return samples


def main() -> None:
    MAX_ROTATION_ANGLE: int = 15
    angle_range = list(range(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE + 1))

    good_dir = Path("good")
    bad_dir = Path("bad")
    good_dir.mkdir(exist_ok=True)
    bad_dir.mkdir(exist_ok=True)

    samples = discover_samples(
        Path("../flux-inpaint/auto_comfyui/results"),
        Path("../flux-inpaint/auto_comfyui/inputs")
    )

    good_count = 0
    bad_count = 0

    with tqdm(total=len(samples), desc="Processing", unit="img") as pbar:
        for img_path, mask_path in samples:
            dest_good = good_dir / Path(img_path).name
            dest_bad = bad_dir / Path(img_path).name

            # Skip processing if already sorted
            if dest_good.exists() or dest_bad.exists():
                # print(f"[SKIP] Already checked: {img_path}")
                if dest_good.exists():
                    good_count += 1
                else:
                    bad_count += 1
                pbar.set_postfix(good=good_count, bad=bad_count, skip=True)
                pbar.update(1)
                continue

            orig = cv2.imread(str(Path(img_path)), cv2.IMREAD_UNCHANGED)
            if orig is None:
                print(f"Image not found: {img_path}")
                continue

            x, y, w, h = get_mask_rect(mask_path)
            crop = orig[y: y + h, x: x + w]

            best_score = -1.0
            best_angle = 0
            best_rot: np.ndarray = crop
            best_proc: np.ndarray = crop
            best_text = ""

            # with ThreadPoolExecutor(max_workers=len(angle_range)) as executor, tqdm(
            #         total=len(angle_range), desc=f"Scanning {img_path}"
            # ) as pbar:
            with ThreadPoolExecutor(max_workers=len(angle_range)) as executor:
                futures = [executor.submit(process_angle, ang, crop) for ang in angle_range]

                for fut in as_completed(futures):
                    score, angle, rot, proc, txt = fut.result()
                    # pbar.update(1)

                    if score > best_score:
                        best_score, best_angle = score, angle
                        best_rot, best_proc, best_text = rot, proc, txt

            if best_score >= 0.5:
                good_count += 1
                target = dest_good
            else:
                bad_count += 1
                target = dest_bad

            shutil.copy2(img_path, target)

            pbar.set_postfix(good=good_count, bad=bad_count)
            pbar.update(1)

            # name_key = "_".join(Path(img_path).name.split('_')[:2])
            # show_pair(orig, (x, y, w, h), best_rot, best_proc, name_key, best_angle)
            # print(f"--- {name_key} -> angle {best_angle}°, score {best_score:.3f} ---")
            # print("Detected text:", best_text)


if __name__ == "__main__":
    main()
