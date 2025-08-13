# import base64
# import json
# from openai import OpenAI
# from PIL import Image
# import io 
# import os
# import logging
# from multiprocessing import Pool, cpu_count
# from functools import partial
# from tqdm import tqdm

# model = 'Qwen/Qwen2.5-VL-32B-Instruct'
# openai_api_key = "123"
# openai_api_base = "http://0.0.0.0:8000/v1"

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # prompt = '''
# # Bạn sẽ được cung cấp:

# # 1. Một ảnh gốc (ảnh 0).
# # 2. Một câu hỏi liên quan đến ảnh gốc.
# # 3. 5 ảnh được cho là có liên quan đến ảnh gốc (ảnh 1 đến 5).

# # Nhiệm vụ của bạn là:
# # - Phân tích ảnh gốc và câu hỏi.
# # - So sánh từng ảnh trong 5 ảnh ứng viên với ảnh gốc và câu hỏi đã cho.
# # - Đưa ra **chỉ số từ 1 đến 5** ứng với ảnh nào có **mức độ liên quan cao nhất** đến ảnh gốc theo nội dung của câu hỏi.

# # Chỉ trả về **duy nhất một số nguyên từ 1 đến 5**, không giải thích.
# # '''
# # prompt = '''
# # Bạn sẽ được cung cấp 2 ảnh.
# # Nhiệm vụ của bạn là cho điểm về độ tương đồng của 2 ảnh này từ 0 đến 10.
# # 0 tương ứng với việc 2 ảnh không liên quan đến nhau, 10 tương ứng với việc 2 ảnh giống nhau.
# # Trả về duy nhất 1 số tự nhiên từ 0 đến 10, không giải thích.
# # '''
# prompt = '''
# Bạn sẽ được cung cấp:
# 1. Một ảnh gốc (ảnh 0)
# 2. Một câu hỏi liên quan đến ảnh gốc.
# 3. Một số ảnh về biển báo được cắt ra từ ảnh 0 (ảnh 1 đến n)

# Nhiệm vụ của bạn là:
# - Chọn ra các ảnh biển báo liên quan đến câu hỏi.
# - Đưa ra danh sách các ảnh liên quan đến câu hỏi.

# Chỉ trả về dạng danh sách "[]" chứa các số nguyên tương ứng với ảnh được chọn.
# '''
# def send_file_request(url):
#     try:
#         if not os.path.exists(url):
#             logger.error(f"File not found: {url}")
#             return None
#         pil_img = Image.open(url)
#         buffered = io.BytesIO()
#         pil_img.save(buffered, format="PNG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         return f"data:image/png;base64,{img_base64}"
#     except Exception as e:
#         logger.error(f"Error processing image {url}: {e}")
#         return None

# def process_image_entry(entry):
#     try:
#         client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

#         for img in entry.get("images", []):
#             img_paths = [img['image_path']] + img['relevant_images']
#             images_encoded = []

#             for url in img_paths:
#                 encoded = send_file_request(url)
#                 if encoded:
#                     images_encoded.append({"type": "image_url", "image_url": {"url": encoded}})
#                 else:
#                     logger.warning(f"Skipping image due to error: {url}")
#                     return entry  # skip if any image fails

#             messages = [{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt + '\n' + entry['question']},
#                     *images_encoded
#                 ]
#             }]

#             response = client.chat.completions.create(
#                 messages=messages,
#                 model=model,
#                 max_tokens=10,
#                 temperature=0,
#             )

#             result = response.choices[0].message.content.strip()
#             img['choose'] = result  # Save result in the image entry

#     except Exception as e:
#         logger.error(f"Error processing entry {entry.get('id', '')}: {e}")
#     return entry

# if __name__ == "__main__":
#     with open('test/submission_1_v1.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     with Pool(processes=4) as pool:
#         results = list(tqdm(pool.imap(process_image_entry, data), total=len(data)))

#     with open('test/submission_1_v1.json', 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
        
        
# import base64
# import json
# from openai import OpenAI
# from PIL import Image
# import io 
# import os
# import logging
# from multiprocessing import Pool
# from tqdm import tqdm

# model = 'Qwen/Qwen2.5-VL-32B-Instruct'
# openai_api_key = "123"
# openai_api_base = "http://0.0.0.0:8000/v1"

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# similarity_prompt = '''
# Bạn sẽ được cung cấp 2 ảnh.
# Nhiệm vụ của bạn là cho điểm về độ tương đồng của 2 ảnh này từ 0 đến 10.
# 0 tương ứng với việc 2 ảnh không liên quan đến nhau, 10 tương ứng với việc 2 ảnh giống nhau.
# Trả về duy nhất 1 số thực từ 0 đến 10, không giải thích.
# '''

# def send_file_request(url):
#     try:
#         if not os.path.exists(url):
#             logger.error(f"File not found: {url}")
#             return None
#         pil_img = Image.open(url)
#         buffered = io.BytesIO()
#         pil_img.save(buffered, format="PNG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         return f"data:image/png;base64,{img_base64}"
#     except Exception as e:
#         logger.error(f"Error processing image {url}: {e}")
#         return None

# def score_similarity(client, image1_path, image2_path):
#     img1 = send_file_request(image1_path)
#     img2 = send_file_request(image2_path)

#     if not img1 or not img2:
#         return None

#     try:
#         response = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": similarity_prompt},
#                         {"type": "image_url", "image_url": {"url": img1}},
#                         {"type": "image_url", "image_url": {"url": img2}}
#                     ]
#                 }
#             ],
#             model=model,
#             max_tokens=5,
#             temperature=0,
#         )
#         score_text = response.choices[0].message.content.strip()
#         return float(score_text)
#     except Exception as e:
#         logger.error(f"Error scoring similarity between {image1_path} and {image2_path}: {e}")
#         return None

# def process_entry(entry):
#     try:
#         client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

#         for img in entry.get("images", []):
#             image_path = img["image_path"]
#             new_relevant_images = []

#             for rel_path in img["relevant_images"]:
#                 score = score_similarity(client, image_path, rel_path)
#                 if score is not None:
#                     new_relevant_images.append({
#                         "image": rel_path,
#                         "score": score
#                     })
#                 else:
#                     logger.warning(f"Failed to score: {rel_path}")
            
#             img["relevant_images"] = new_relevant_images

#     except Exception as e:
#         logger.error(f"Error processing entry {entry.get('id', '')}: {e}")
#     return entry

# if __name__ == "__main__":
#     with open('test/submission_1_v1.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     with Pool(processes=8) as pool:
#         results = list(tqdm(pool.imap(process_entry, data), total=len(data)))

#     with open('caption_test_data_crop.json', 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)




# import base64
# import json
# from openai import OpenAI
# from PIL import Image
# import io
# import os
# import logging
# from multiprocessing import Pool
# from tqdm import tqdm

# model = 'Qwen/Qwen2.5-VL-32B-Instruct'
# openai_api_key = "123"
# openai_api_base = "http://0.0.0.0:8000/v1"

# ROOT_IMAGE_DIR = "/home/jovyan/chien/vlsp/VLSP 2025 - MLQA-TSR Data Release/test/public_test_images"

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# prompt = '''
# Bạn sẽ được cung cấp:
# 1. Một ảnh gốc (ảnh 0)
# 2. Một câu hỏi liên quan đến ảnh gốc.
# 3. Một số ảnh về biển báo được cho là liên quan ảnh gốc (ảnh 1 đến n)

# Nhiệm vụ của bạn là:
# - Chọn ra các ảnh biển báo liên quan đến câu hỏi.

# Chỉ trả về dạng danh sách "[]" chứa các số nguyên tương ứng với ảnh được chọn, không giải thích
# '''

# def send_file_request(url):
#     try:
#         if not os.path.exists(url):
#             logger.error(f"File not found: {url}")
#             return None
#         pil_img = Image.open(url).convert("RGB")
#         buffered = io.BytesIO()
#         pil_img.save(buffered, format="PNG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         return f"data:image/png;base64,{img_base64}"
#     except Exception as e:
#         logger.error(f"Error processing image {url}: {e}")
#         return None

# def process_image_entry(entry):
#     try:
#         client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

#         # Load root image (ảnh 0)
#         root_image_path = os.path.join(ROOT_IMAGE_DIR, f"{entry['image_id']}.jpg")
#         root_image_encoded = send_file_request(root_image_path)
#         if root_image_encoded is None:
#             logger.warning(f"Skipping entry {entry['id']} due to missing root image")
#             return entry

#         # Load relevant_images của từng biển báo (ảnh 1 → n)
#         image_encodings = []
#         mapping_index_to_path = []  # để sau này biết ảnh nào được chọn

#         for img in entry["images"]:
#             for rel in img.get("relevant_images", []):
#                 url = rel["image"]
#                 encoded = send_file_request(url)
#                 if encoded:
#                     image_encodings.append({"type": "image_url", "image_url": {"url": encoded}})
#                     mapping_index_to_path.append(url)
#                 else:
#                     logger.warning(f"Could not load relevant image: {url}")

#         if not image_encodings:
#             logger.warning(f"No relevant images for entry {entry['id']}")
#             entry["selected_image"] = None
#             return entry

#         # Compose messages
#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt + '\n' + entry["question"]},
#                 {"type": "image_url", "image_url": {"url": root_image_encoded}},
#                 *image_encodings
#             ]
#         }]

#         response = client.chat.completions.create(
#             messages=messages,
#             model=model,
#             max_tokens=20,
#             temperature=0,
#         )

#         result = response.choices[0].message.content
#         entry["selected_image"] = result
# #         try:
# #             selected_indices = json.loads(result)
# #             if isinstance(selected_indices, list) and len(selected_indices) > 0:
# #                 # Lấy ảnh đầu tiên trong danh sách được chọn
# #                 chosen_index = selected_indices[0]
# #                 entry["selected_image"] = mapping_index_to_path[chosen_index]
# #             else:
# #                 entry["selected_image"] = None
# #         except Exception as e:
# #             logger.error(f"Failed to parse result: {result} | Error: {e}")
# #             entry["selected_image"] = None

#     except Exception as e:
#         logger.error(f"Error processing entry {entry.get('id', '')}: {e}")
#         entry["selected_image"] = None

#     return entry

# if __name__ == "__main__":
#     with open('caption_test_data_crop_filtered.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     with Pool(processes=2) as pool:
#         results = list(tqdm(pool.imap(process_image_entry, data), total=len(data)))

#     with open('caption_test_data_crop_filtered_selected_v1.json', 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)




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
    INPUT = 'caption_test_data_crop_image_rettrieve_v1.json'
    OUTPUT = 'caption_test_data_crop_image_rettrieve_v1_filter.json'

    with open(INPUT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Tùy GPU/CPU có thể tăng processes
    with Pool(processes=2) as pool:
        results = list(tqdm(pool.imap(process_image_entry, data), total=len(data)))

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Done. Wrote: {OUTPUT}")
