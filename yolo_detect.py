import numpy as np
import requests
from ultralytics import YOLO
import cv2


def read_remote_image(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise Exception(f"下载图片失败: {str(e)}")


if __name__ == "__main__":
    img_ctx = read_remote_image("http://oss.fyxsyz.cn/2025102811003899948.jpg")
    image_array = np.asarray(bytearray(img_ctx), dtype=np.uint8)
    # 解码为OpenCV图像
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    onnx_model = YOLO("runs/detect/train/weights/best.onnx")
    results = onnx_model(img)
    print("模型类别:", onnx_model.names)

    # 定义颜色映射，类别0用绿色，类别1用红色
    colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # 绿色 (BGR格式)  # 红色 (BGR格式)

    # 遍历所有检测结果
    for result in results:
        # 获取原始图像e
        img = result.orig_img.copy()  # 复制原图避免修改原始数据
        boxes = result.boxes

        if boxes is not None:
            for i in range(len(boxes)):
                # 提取坐标、置信度、类别ID
                xyxy = boxes.xyxy[i].cpu().numpy()  # 边界框坐标 [x1, y1, x2, y2]
                conf = boxes.conf[i].cpu().item()  # 置信度
                cls_id = int(boxes.cls[i].cpu().item())  # 类别ID
                label = result.names[cls_id]  # 类别名称

                # 获取对应类别的颜色，默认为蓝色
                color = colors.get(cls_id, (255, 0, 0))

                # 绘制边界框
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # 添加标签文本
                text = f"{label} {conf:.2f}"
                # 计算文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                # 绘制文本背景框
                cv2.rectangle(
                    img,
                    (x1, y1 - text_height - baseline),
                    (x1 + text_width, y1),
                    color,
                    -1,
                )
                # 绘制文本
                cv2.putText(
                    img,
                    text,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                print(f"检测到 {label}，置信度: {conf:.2f}，位置: {xyxy}")

        # 如果需要保存图像，取消下面这行注释
        cv2.imwrite("result.jpg", img)
