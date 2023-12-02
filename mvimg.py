import json
import shutil
import os

# 加载 JSON 文件
with open('/home/lvheng_bishe/data/humancar/datasets/annotations/val.json') as f:
    data = json.load(f)

# 设置原始图像和目标训练图像文件夹的路径
original_images_path = '/home/lvheng_bishe/data/humancar/datasets/images'
train_images_path = '/home/lvheng_bishe/data/humancar/datasets/val/images'
os.makedirs(train_images_path, exist_ok=True)

# 遍历 train.json 中的所有图像，并将它们复制到训练文件夹
for image in data['images']:
    file_name = image['file_name']
    original_file_path = os.path.join(original_images_path, file_name)
    target_file_path = os.path.join(train_images_path, file_name)

    # 复制文件
    shutil.copy2(original_file_path, target_file_path)

print("图像复制完成")
