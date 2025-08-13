import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# === Load data ===
with open("test/submission_1.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("caption_db_crop_v1.json", "r", encoding="utf-8") as f:
    db_data = json.load(f)

# === Load embedding model ===
model = SentenceTransformer("BAAI/bge-m3")

# === Extract captions from DB ===
db_captions = [item["caption"] for item in db_data]
db_images = [item["image"] for item in db_data]

# Encode DB embeddings once
db_embeddings = model.encode(db_captions, convert_to_tensor=True)

# Parameters
top_k = 5
output_data = []

# === Process each train sample ===
for item in tqdm(train_data, desc="Processing queries"):
    new_item = dict(item)
    image_caption_list = item['images']

    # Prepare new image_caption list with relevant images
    updated_captions = []

    for crop in image_caption_list:
        # crop: { "path": "caption text" }
        img_path = crop['image']
        caption = crop['caption']

        # Encode this crop caption
        query_embedding = model.encode(caption, prompt_name="query", convert_to_tensor=True)

        # Compute similarity
        scores = util.cos_sim(query_embedding, db_embeddings)[0]

        # Get top-k results
        top_results = torch.topk(scores, k=top_k)
        relevant_images = [db_images[idx] for idx in top_results.indices]

        # Append new structure
        updated_captions.append({
            "image_path": img_path,
            "caption": caption,
            "relevant_images": relevant_images
        })

    # Replace image_caption with updated structure
    new_item["images"] = updated_captions
    output_data.append(new_item)

# Save result
with open("test/submission_1_v1.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("✅ Done. File saved to train_data/vlsp_2025_train_add_caption_with_relevant.json")
