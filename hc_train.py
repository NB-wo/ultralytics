from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8x.pt")  # 选择合适的模型，例如 yolov8n

# 设置训练参数
data_config = "mydata.yaml"  # 数据集配置文件路径
epochs = 400  # 训练轮次
img_size = 1280  # 图像大小，按照您的需求设置

# 训练模型
model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=12 ,device=[0,1,3,4])

print("训练完成")
