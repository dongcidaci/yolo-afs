# 示例：使用 Ultralytics 导出 YOLOv8/v10 模型为 ONNX
from ultralytics import YOLO

# 这里注意修改路径，没次训练的目录不同
model = YOLO('runs/detect/train/weights/best.pt')
model.export(format='onnx', dynamic=True, simplify=True)  # 生成 best.onnx