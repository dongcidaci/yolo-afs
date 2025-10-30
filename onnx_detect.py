import cv2
import numpy as np
import onnxruntime as ort
import requests
import json


def read_remote_image(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise Exception(f"下载图片失败: {str(e)}")


class SimpleYOLODetector:
    def __init__(self, model_path, conf_threshold=0.1, nms_threshold=0.1):
        """
        初始化简化版YOLO检测器
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 获取输入形状并确保是数值类型
        input_shape = self.session.get_inputs()[0].shape
        try:
            self.input_height = int(input_shape[2]) if len(input_shape) > 2 else 640
            self.input_width = int(input_shape[3]) if len(input_shape) > 3 else 640
        except (ValueError, TypeError):
            # 默认值 fallback
            self.input_height = 640
            self.input_width = 640

    def has_detections(self, image_path, visualize=False):
        """
        检测图片中是否有目标

        Args:
            image_path: 图像路径（支持本地路径和URL）
            visualize: 是否保存带标注的结果图

        Returns:
            list: 包含每个检测目标详细信息的字典列表
        """
        # 判断是否为URL
        if image_path.startswith(("http://", "https://")):
            # 使用您提供的方法读取远程图片
            try:
                image_data = read_remote_image(image_path)
                # 将字节数据转换为numpy数组并解码
                np_arr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"无法解码图像: {image_path}")
            except Exception as e:
                raise Exception(f"加载远程图像失败: {str(e)}")
        else:
            # 读取本地图像
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {image_path}")

        original_image = image.copy()

        # 预处理
        input_tensor, ratio, pad = self._preprocess_with_params(image)

        # 执行推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})

        # 解析结果
        detections = self._parse_detections(outputs, ratio, pad, original_image.shape)

        # 可视化（可选）
        if visualize and len(detections) > 0:
            result_image = self._draw_detections(original_image.copy(), detections)
            cv2.imwrite("detection_result.jpg", result_image)

        return detections

    def _draw_detections(self, image, detections):
        """
        在图像上绘制检测框（按类别着色）

        Args:
            image: 原始图像
            detections: 检测结果列表

        Returns:
            image: 绘制了检测框的图像
        """
        try:
            h, w = image.shape[:2]

            # 类别颜色映射：类别0用绿色，类别1用红色
            colors = {
                0: (0, 255, 0),  # 绿色 - afs_box
                1: (0, 0, 255),  # 红色 - afs_stamp
            }

            for detection in detections:
                class_id = detection["class_id"]
                confidence = detection["confidence"]
                bbox = detection["bbox"]

                # 获取对应类别的颜色
                color = colors.get(class_id, (255, 255, 255))  # 默认白色

                # 转换边界框坐标到像素位置
                x_center = bbox["x_center"]
                y_center = bbox["y_center"]
                box_w = bbox["width"]
                box_h = bbox["height"]

                # 计算边界框的左上角和右下角坐标
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)

                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # 添加标签（类别和置信度）
                label = f"{class_id}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            return image
        except Exception as e:
            print(f"绘制检测框时出错: {e}")
            return image

    def _preprocess_with_params(self, image):
        """
        图像预处理，返回变换参数
        """
        # 使用 letterbox 保持宽高比
        letterboxed, ratio, pad = letterbox(
            image, (self.input_height, self.input_width), auto=False
        )
        input_image = letterboxed.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image, ratio, pad

    def _parse_detections(self, outputs, ratio, pad, orig_shape):
        """
        解析检测结果，返回详细的目标信息，并将坐标还原到原图
        """
        detections = []

        try:
            predictions = np.squeeze(outputs[0])  # (84, 8400)
            if len(predictions.shape) == 3:
                predictions = np.squeeze(predictions, axis=0)
            if predictions.shape[0] == 84:  # (84, 8400) -> (8400, 84)
                predictions = predictions.T

            # 分离
            boxes = predictions[:, :4]  # cx, cy, w, h
            obj_logits = predictions[:, 4]  # objectness logit
            cls_logits = predictions[:, 5:]  # class logits

            # 只保留有效的类别（根据你的dataset.yaml，应只有2个类别）
            cls_logits = cls_logits[:, :2]  # 限制为2个类别

            # 应用 sigmoid 获取概率
            obj_probs = 1 / (1 + np.exp(-obj_logits))
            cls_probs = 1 / (1 + np.exp(-cls_logits))

            # 计算最终置信度 = objectness * class_probability
            max_cls_probs = np.max(cls_probs, axis=1)
            final_scores = obj_probs * max_cls_probs
            class_ids = np.argmax(cls_probs, axis=1)

            # 过滤低置信度检测
            valid_indices = final_scores > self.conf_threshold

            # 获取原图尺寸
            orig_h, orig_w = orig_shape[:2]

            # 获取letterbox参数
            gain = min(ratio[0], ratio[1])  # 缩放比例
            pad_x, pad_y = pad

            for i in np.where(valid_indices)[0]:
                # 获取网络输出的原始坐标（相对于网络输入尺寸）
                x_center = boxes[i][0]
                y_center = boxes[i][1]
                box_w = boxes[i][2]
                box_h = boxes[i][3]

                # 将坐标从网络输入尺寸还原到原图尺寸
                # 注意：网络输出坐标是相对于网络输入尺寸(通常是640x640)的
                # 需要先减去填充，再除以缩放比例
                x_center = (x_center - pad_x) / gain
                y_center = (y_center - pad_y) / gain
                box_w = box_w / gain
                box_h = box_h / gain

                # 确保坐标在原图范围内
                x_center = np.clip(x_center, 0, orig_w)
                y_center = np.clip(y_center, 0, orig_h)
                box_w = np.clip(box_w, 0, orig_w)
                box_h = np.clip(box_h, 0, orig_h)

                detection = {
                    "id": len(detections),
                    "class_id": int(class_ids[i]),
                    "confidence": float(final_scores[i]),
                    "bbox": {
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width": float(box_w),
                        "height": float(box_h),
                    },
                }
                detections.append(detection)

            return detections
        except Exception as e:
            print(f"解析出错: {e}")
            return []


# 使用示例
def check_image_for_objects(model_path, image_path):
    """
    检查图像中是否包含可检测的目标，并返回详细信息

    Args:
        model_path: ONNX模型路径
        image_path: 待检测图像路径

    Returns:
        list: 包含每个检测目标详细信息的字典列表
    """
    try:
        detector = SimpleYOLODetector(model_path)
        detections = detector.has_detections(image_path, visualize=True)
        return detections
    except Exception as e:
        print(f"检测过程出错: {e}")
        return []


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    将图像按比例缩放并填充至指定尺寸（保持宽高比）

    Args:
        img: 输入图像 (HWC, BGR)
        new_shape: 目标尺寸 (height, width) 或单值 (如 640)
        color: 填充颜色 (B, G, R)
        auto: 是否调整尺寸为 stride 的倍数（用于 TensorRT 等）
        scaleFill: 是否直接拉伸填充（不保持比例）
        scaleup: 是否允许放大图像（False 则只缩小）
        stride: 最小下采样步长（通常为 32）

    Returns:
        img: letterbox 后的图像
        ratio: (w_ratio, h_ratio) 缩放比例
        pad: (w_pad, h_pad) 填充像素数（左/上, 右/下）
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 获取原始尺寸
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大
        r = min(r, 1.0)

    # 计算 padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # 最小化矩形（使尺寸为 stride 的倍数）
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # 拉伸填充
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # 均匀填充（左右/上下各一半）
    dw /= 2
    dh /= 2

    # 缩放图像
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 添加填充
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return img, ratio, (left, top)  # 返回图像、缩放比例、左上角 padding


# 示例调用
if __name__ == "__main__":
    model_file = "runs/detect/train/weights/best.onnx"
    test_image = "http://oss.fyxsyz.cn/2025102811118645754.jpeg"

    detections = check_image_for_objects(model_file, test_image)

    print(f"检测到 {len(detections)} 个目标:")
    for detection in detections:
        print(
            f"  目标 {detection['id']}: 类别={detection['class_id']}, "
            f"置信度={detection['confidence']:.3f}, "
            f"边界框={detection['bbox']}"
        )

    # 输出为格式化的JSON
    print("\nJSON格式输出:")
    print(json.dumps(detections, indent=2, ensure_ascii=False))
