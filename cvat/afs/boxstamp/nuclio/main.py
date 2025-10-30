import base64
import json
import logging
import numpy as np
import cv2
from ultralytics import YOLO

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 类别名称映射（必须与 function.yaml 中 spec 的 name 顺序一致！）
# YOLO class index -> label name
CLASS_NAMES = ["afs_box", "afs_stamp"]  # index 0 -> afs_box, index 1 -> afs_stamp


def init_context(context):
    """
    初始化上下文，仅在函数启动时调用一次。
    这里可以放置一些全局变量和资源的初始化工作，比如模型加载等。
    """
    logger.info("开始初始化上下文...")

    # 加载模型
    try:
        MODEL_PATH = "/opt/nuclio/afs/boxstamp/nuclio/best.pt"
        context.user_data.model_handler = YOLO(MODEL_PATH)
        logger.info(f"✅ YOLO 模型加载成功: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


def handler(context, event):
    """
    CVAT Auto Annotation Detector Handler
    Input: {"image": "<base64>"}
    Output: [{"label": "afs_box", "confidence": "0.95", "points": [x1,y1,x2,y2], "type": "rectangle"}, ...]
    """
    try:
        # 解析请求体
        body = event.body
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        data = json.loads(body)
        image_b64 = data.get("image")
        if not image_b64:
            return context.Response(
                body=json.dumps({"error": "缺少 'image' 字段（base64 编码）"}),
                headers={"Content-Type": "application/json"},
                status_code=400,
            )

        # 解码 base64 图像
        try:
            image_data = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV 无法解码图像")
        except Exception as e:
            logger.error(f"图像解码失败: {e}")
            return context.Response(
                body=json.dumps({"error": f"图像解码失败: {str(e)}"}),
                headers={"Content-Type": "application/json"},
                status_code=400,
            )

        # 推理
        model = context.user_data.model_handler
        results = model(img, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float).tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # 获取标签名
                if cls_id >= len(CLASS_NAMES):
                    logger.warning(f"检测到未知类别 ID: {cls_id}，跳过")
                    continue
                label_name = CLASS_NAMES[cls_id]

                detections.append(
                    {
                        "label": label_name,
                        "confidence": str(conf),  # 必须是字符串！
                        "points": [x1, y1, x2, y2],
                        "type": "rectangle",
                    }
                )

        logger.info(f"✅ 推理完成，检测到 {len(detections)} 个目标")
        return context.Response(
            body=json.dumps(detections),
            headers={"Content-Type": "application/json"},
            status_code=200,
        )

    except Exception as e:
        logger.exception("处理请求时发生异常")
        return context.Response(
            body=json.dumps({"error": f"内部错误: {str(e)}"}),
            headers={"Content-Type": "application/json"},
            status_code=500,
        )
