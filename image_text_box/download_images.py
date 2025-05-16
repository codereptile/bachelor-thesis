from pathlib import Path
from tqdm import tqdm
import requests

def download_images(txt_path: str, output_dir: str, base_name: str) -> None:
    txt_file = Path(txt_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = txt_file.read_text(encoding='utf-8').splitlines()
    width = len(str(len(lines)))
    for idx, url in enumerate(tqdm(lines, desc="Downloading images", unit="image")):
        number = idx + 1
        output_file = output_path / f"{base_name}_{number:0{width}d}.jpg"
        if output_file.exists():
            continue
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            output_file.write_bytes(response.content)
        except Exception as e:
            print(e)
            continue

NAME = "hotel"

download_images(f"{NAME}s.txt", f"{NAME}s", NAME)
