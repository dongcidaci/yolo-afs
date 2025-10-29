from ultralytics import YOLO

onnx_model = YOLO("runs/detect/train/weights/best.onnx")

results = onnx_model("http://oss.fyxsyz.cn/2025102811024288088.jpg")
print(results)
