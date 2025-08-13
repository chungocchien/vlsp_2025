import base64
import json
from openai import OpenAI
from PIL import Image
import io
import os
import logging
from multiprocessing import Pool
from tqdm import tqdm

# ===== Config =====
model = 'Qwen/Qwen2.5-VL-32B-Instruct'
openai_api_key = "123"
openai_api_base = "http://0.0.0.0:8000/v1"

# Ảnh gốc (image 0) nằm theo image_id.jpg tại thư mục này
ROOT_IMAGE_DIR = "/home/jovyan/chien/vlsp/VLSP 2025 - MLQA-TSR Data Release/test_v1/public_test_infer"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt = """You will be provided with:
- The original image (image 0)
- A question about the original image
- ONE cropped traffic-sign image cut from the original image (+ its caption)

Task:
Decide if THIS cropped image is relevant to answering the question about the original image.

Return only "yes" or "no". No explanation.
"""

def send_file_request(url: str):
    try:
        if not os.path.exists(url):
            logger.error(f"File not found: {url}")
            return None
        pil_img = Image.open(url).convert("RGB")
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error processing image {url}: {e}")
        return None

def ask_is_relevant(client: OpenAI, question: str, root_image_encoded: str, crop_image_path: str, crop_caption: str):
    """Gọi model để đánh giá 1 cropped image."""
    crop_image_encoded = send_file_request(crop_image_path)
    if crop_image_encoded is None:
        return None  # không đánh giá được

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt + "\nQuestion: " + question + "\nCropped image caption: " + (crop_caption or "")},
            {"type": "image_url", "image_url": {"url": root_image_encoded}},   # original image (image 0)
            {"type": "image_url", "image_url": {"url": crop_image_encoded}},   # the cropped image to evaluate
        ]
    }]

    resp = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=4,
        temperature=0,
    )
    ans = (resp.choices[0].message.content or "").strip().lower()
    # Chuẩn hóa về yes/no
    if ans.startswith("y"):
        return "yes"
    if ans.startswith("n"):
        return "no"
    # fallback: anything else -> no
    return "no"

def process_image_entry(entry: dict):
    """Xử lý 1 item dữ liệu:
    - Load ảnh gốc theo image_id
    - Lặp qua từng phần tử trong images_inside và thêm is_relevant + model_raw
    """
    try:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

        # Load root image (image 0)
        root_image_path = os.path.join(ROOT_IMAGE_DIR, f"{entry['image_id']}.jpg")
        root_image_encoded = send_file_request(root_image_path)
        if root_image_encoded is None:
            logger.warning(f"Skipping entry {entry.get('id')} due to missing root image")
            return entry

        images_inside = entry.get("images_inside", [])
        if not isinstance(images_inside, list) or len(images_inside) == 0:
            logger.warning(f"No images_inside for entry {entry.get('id')}")
            return entry

        question = entry.get("question", "")

        # Đánh giá từng cropped image
        for img in images_inside:
            crop_path = img.get("image")
            crop_caption = img.get("caption", "")
            if not crop_path:
                img["is_relevant"] = False
                img["model_raw"] = "no"
                continue

            try:
                ans = ask_is_relevant(client, question, root_image_encoded, crop_path, crop_caption)
                if ans is None:
                    img["is_relevant"] = False
                    img["model_raw"] = "no"
                else:
                    img["is_relevant"] = (ans == "yes")
                    img["model_raw"] = ans
            except Exception as e:
                logger.error(f"Model error on {entry.get('id')} - {crop_path}: {e}")
                img["is_relevant"] = False
                img["model_raw"] = "no"

        return entry

    except Exception as e:
        logger.error(f"Error processing entry {entry.get('id', '')}: {e}")
        return entry

if __name__ == "__main__":
    INPUT = 'caption_test_data_choose_image_crop.json'
    OUTPUT = 'caption_test_data_choose_image_crop_filter.json'

    with open(INPUT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Tùy GPU/CPU có thể tăng processes
    with Pool(processes=2) as pool:
        results = list(tqdm(pool.imap(process_image_entry, data), total=len(data)))

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Done. Wrote: {OUTPUT}")
