import json
import os
from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载预训练模型
model = YOLO("yolov8l.pt")

# 设置数据集路径
dataset_path = "/home/lvheng_bishe/data/humancar/datasets"
test_images_path = os.path.join(dataset_path, "images")
annotations_path = os.path.join(dataset_path, "annotations", "test.json")

# 读取测试集图像信息
with open(annotations_path) as file:
    test_data = json.load(file)

# 创建结果文件夹
results_path = "results"

# 对测试集图像进行预测
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
                # 提取坐标和类别
                x1, y1, x2, y2 = boxes.xyxy[i]
                category_id = boxes.cls[i].item()

                # 保存格式：类别:左上角x, 左上角y, 右下角x, 右下角y
                result_file.write(f"{category_id}:{x1},{y1},{x2},{y2}\n")

print("预测完成，结果已保存。")