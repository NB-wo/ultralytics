import json
import os

# 加载 JSON 文件
with open('/home/lvheng_bishe/data/humancar/datasets/annotations/test.json') as f:
    data = json.load(f)

# 创建一个字典，将image_id映射到文件名
image_id_to_file_name = {image['id']: image['file_name'] for image in data['images']}

# 处理每个标注
for ann in data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id'] - 1  # 减1以从0开始计数
    bbox = ann['bbox']
    
    # 获取图像尺寸以进行归一化
    file_name = image_id_to_file_name[image_id]
    image_width = next(item for item in data["images"] if item["id"] == image_id)["width"]
    image_height = next(item for item in data["images"] if item["id"] == image_id)["height"]

    # 计算归一化的中心坐标和宽高
    x_center = (bbox[0] + bbox[2] / 2) / image_width
    y_center = (bbox[1] + bbox[3] / 2) / image_height
    width = bbox[2] / image_width
    height = bbox[3] / image_height

    # 创建或追加到txt文件
    txt_path = os.path.join('/home/lvheng_bishe/data/humancar/datasets/test/labels/', os.path.splitext(file_name)[0] + '.txt')
    with open(txt_path, 'a') as f:
        f.write(f'{category_id} {x_center} {y_center} {width} {height}\n')
    
    print("1\n")

print("转换完成")
