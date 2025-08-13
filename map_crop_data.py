import json

with open("caption_test_data_choose_image_crop.json", "r") as f:
    data = json.load(f)
with open("caption_test_crop.json", "r") as f:
    cap_data = json.load(f)
for d in data:
    d['images_inside'] = []
    for c in cap_data:
        if d['image_id'] in c['image']:
            d['images_inside'].append(c)
with open("caption_test_data_choose_image_crop.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
