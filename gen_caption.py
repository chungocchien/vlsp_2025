import base64
import json
from openai import OpenAI
from PIL import Image
import io 
import os
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

model = 'Qwen/Qwen2.5-VL-32B-Instruct'

openai_api_key = "123"
openai_api_base = "http://0.0.0.0:8000/v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_file_request(url):
    """Convert image file to base64 encoded string"""
    try:
        if not os.path.exists(url):
            logger.error(f"File not found: {url}")
            return None
        pil_img = Image.open(url)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error processing image {url}: {e}")
        return None

def run_single_image(d):
    """Process a single image with the given query"""
    url = d['image']
    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": '<image>Hãy mô tả đơn giản biển báo giao thông trong ảnh. Hãy cho biết loại biển báo, hình dạng, màu sắc. Trả lời ngắn gọn, khoảng 1-2 câu.'},
                        *[
                            {"type": "image_url", "image_url": {"url": send_file_request(url)}}
                        ]
                    ]
                }
            ],
            model=model,
            max_tokens=1024,
            temperature=0,
        )
        d['caption'] = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing image {url}: {str(e)}")
        d['caption'] = f"Error: {str(e)}"
    return d

if __name__ == "__main__":
    with open('caption_test_crop.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with Pool(processes=2) as pool:
        results = list(tqdm(pool.imap(run_single_image, data), total=len(data)))

    with open('caption_test_crop.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)