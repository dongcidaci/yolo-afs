from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # pass any model type
results = model.train(epochs=5)