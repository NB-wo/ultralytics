import json
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载预训练模型
model = YOLO("runs/detect/train/weights/best.pt")

# 设置数据集路径
dataset_path = "/home/lvheng_bishe/data/humancar/datasets"
test_images_path = os.path.join(dataset_path, "images")
annotations_path = os.path.join(dataset_path, "annotations", "test.json")

# 读取测试集图像信息
with open(annotations_path) as file:
    test_data = json.load(file)

# 创建结果文件夹
results_path = "submission"
os.makedirs(results_path, exist_ok=True)

# 预测所有测试集图像并生成txt文件
for image_info in test_data["images"]:
    image_path = os.path.join(test_images_path, image_info["file_name"])
    results = model(image_path)

    # 转换预测结果并保存到txt文件
    file_name_without_extension = os.path.splitext(image_info["file_name"])[0]
    result_file_path = os.path.join(results_path, f"{file_name_without_extension}.txt")
    with open(result_file_path, "w") as result_file:
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                # 提取坐标、类别和置信度，并转换为整数
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                category_id = int(boxes.cls[i].item() + 1)
                confidence = round(boxes.conf[i].item(), 2)  # 置信度保留两位小数

                # 保存格式：类别:左上角x 左上角y 右下角x 右下角y 置信度
                result_file.write(f"{category_id}:{x1} {y1} {x2} {y2} {confidence}\n")
'''
# 对前5张图像生成bbox以及标注类别
annotated_images_path = "annotated_images"
for image_info in test_data["images"][:5]:
    image_path = os.path.join(test_images_path, image_info["file_name"])
    results = model(image_path)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=16)

    # 绘制边界框和类别
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i]
            category_id = int(boxes.cls[i].item() + 1)  # 根据需要调整类别ID
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 4, y1 + 4), f"{category_id}", fill="red", font=font)

    # 保存绘制了bbox的图像
    file_name_without_extension = os.path.splitext(image_info["file_name"])[0]
    annotated_image_file = f"{file_name_without_extension}_annotated.jpg"
    annotated_image_path = os.path.join(annotated_images_path, annotated_image_file)
    img.save(annotated_image_path)

print("预测完成，结果已保存。所有图像的txt文件已生成，前5张图像的标注图像已保存。")
'''