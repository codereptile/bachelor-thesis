from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Any, Dict
import requests
import websocket
import os
from tqdm import tqdm

def run_comfyui_workflow(
    image_path: str,
    output_dir: str,
    seed: int,
    steps: int,
    prompt: str,
    workflow_path: str = "main_test_auto_images_api.json",
    server: str = "127.0.0.1:8188",
) -> str:
    def load_workflow(path: str) -> Dict[str, Any]:
        return json.loads(Path(path).read_text())

    def override_workflow(
        data: Dict[str, Any],
        image_name: str,
        seed: int,
        steps: int,
        prompt_text: str,
    ) -> Dict[str, Any]:
        for node_id, node in data.items():
            if node["class_type"] == "LoadImage":
                node["inputs"]["image"] = image_name
                node["inputs"]["upload"] = "image"
            elif node["class_type"] == "RandomNoise":
                node["inputs"]["noise_seed"] = seed
            elif node["class_type"] == "BasicScheduler":
                node["inputs"]["steps"] = steps
            elif node_id == "35":
                node["inputs"]["text_a"] = prompt_text
        return data

    def upload_image(path: str, server: str) -> str:
        file_name = Path(path).name
        with open(path, "rb") as f:
            requests.post(f"http://{server}/upload/image", files={"image": (file_name, f, "image/png")}).raise_for_status()
        return file_name

    def queue_prompt(prompt: Dict[str, Any], client_id: str, server: str) -> str:
        r = requests.post(f"http://{server}/prompt", json={"prompt": prompt, "client_id": client_id})
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait_for_done(prompt_id: str, client_id: str, server: str) -> Dict[str, Any]:
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server}/ws?clientId={client_id}")
        try:
            while True:
                message = json.loads(ws.recv())
                if message["type"] == "executing" and message["data"]["node"] is None and message["data"]["prompt_id"] == prompt_id:
                    break
        finally:
            ws.close()
        return requests.get(f"http://{server}/history/{prompt_id}").json()

    def download_main_output(history: Dict[str, Any], prompt_id: str, save_path: Path, server: str) -> None:
        outputs = history[prompt_id]["outputs"]
        for node in outputs.values():
            if "images" in node:
                for img in node["images"]:
                    if img["filename"].startswith("pl_"):
                        params = {"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]}
                        data = requests.get(f"http://{server}/view", params=params)
                        data.raise_for_status()
                        save_path.write_bytes(data.content)
                        return
        raise RuntimeError("Main output image not found")

    workflow = load_workflow(workflow_path)
    uploaded_name = upload_image(image_path, server)
    patched_workflow = override_workflow(workflow, uploaded_name, seed, steps, prompt)
    client_id = str(uuid.uuid4())
    prompt_id = queue_prompt(patched_workflow, client_id, server)
    history = wait_for_done(prompt_id, client_id, server)

    input_stem = Path(image_path).stem
    final_name = f"{input_stem}_seed{seed}_steps{steps}.png"
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / final_name

    download_main_output(history, prompt_id, output_path, server)
    return str(output_path)

def process_directory(
    input_dir: str,
    output_dir: str,
    seed: int,
    steps: int,
    prompt: str,
    workflow_path: str = "main_test_auto_images_api.json",
    server: str = "127.0.0.1:8188",
) -> None:
    input_path = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for file_path in tqdm(sorted(input_path.iterdir(), key=lambda p: p.name), desc="Processing files"):
        if file_path.is_file():
            input_stem = file_path.stem
            final_name = f"{input_stem}_seed{seed}_steps{steps}.png"
            output_file = Path(output_dir) / final_name
            if output_file.exists():
                # print(f"Skipping {file_path.name}, output already exists: {output_file}")
                continue
            output = run_comfyui_workflow(
                image_path=str(file_path),
                output_dir=output_dir,
                seed=seed,
                steps=steps,
                prompt=prompt,
                workflow_path=workflow_path,
                server=server,
            )

if __name__ == "__main__":
    process_directory(
        input_dir="inputs",
        output_dir="results2",
        seed=2,
        steps=15,
        prompt='Facade of a shop with a large clear text saying: "Astronomia" in dark letters.'
    )
