from pathlib import Path

from transformers import pipeline

pipe = pipeline('image-classification', model="prithivMLmods/Deep-Fake-Detector-v2-Model", device="cpu")

def check_image(img_path: str) -> float:
    result = pipe(img_path)
    scores = {}
    for r in result:
        scores[r['label']] = r['score']

    fake = scores['Deepfake']
    real = scores['Realism']
    print(f"{img_path:80} Fake: {fake:.3f} Real: {real:.3f} Normalized fake: {fake/real:.3f}")
    return fake/real

# Predict on an image
check_image("flux_detection_reference_image_01.jpg")
check_image("flux_detection_reference_image_02.jpg")
check_image("flux_detection_reference_image_03.jpg")
check_image("flux_detection_reference_image_04.jpg")
check_image("flux_detection_reference_image_05.jpg")
check_image("stars_billboard_source_image.jpg")
check_image("stars_billboard_best_output.png")

CHECK_FIRST_N = 50

total = 0
count = 0

i = 0
for file_path in Path("../flux-inpaint/auto_comfyui/results").iterdir():
    i += 1
    if i > CHECK_FIRST_N:
        break
    if file_path.is_file():
        total += check_image(str(file_path))
        count += 1
        
print(f"Average: {total/count:.3f}")


total = 0
count = 0

i = 0
for file_path in Path("../image_text_box/shops").iterdir():
    i += 1
    if i > CHECK_FIRST_N:
        break
    if file_path.is_file():
        total += check_image(str(file_path))
        count += 1
        
print(f"Average: {total/count:.3f}")
