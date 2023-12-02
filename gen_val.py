import json

# 加载原始 train.json 文件
with open('/home/lvheng_bishe/data/humancar/datasets/annotations/train.json') as f:
    data = json.load(f)

# 选出前 100 个标注
selected_annotations = data['annotations'][:100]
selected_image_ids = {ann['image_id'] for ann in selected_annotations}

# 根据选出的标注提取对应的图像
selected_images = [img for img in data['images'] if img['id'] in selected_image_ids]

val_data = {
    'images': selected_images,
    'annotations': selected_annotations,
    'categories': data['categories']
}

# 将这些标注保存到新的 JSON 文件中
with open('/home/lvheng_bishe/data/humancar/datasets/annotations/val.json', 'w') as f:
    json.dump(val_data, f)

print("已创建验证集 val.json")
