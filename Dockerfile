# 使用 ultralytics/ultralytics 作为基础镜像，因为它已经包含了 YOLO 和相关依赖
FROM ultralytics/ultralytics:latest

# 设置工作目录
WORKDIR /opt/nuclio

# 将当前目录下的所有文件复制到容器内的 /opt/nuclio 目录中
COPY ./cvat .

# 安装任何额外的 Python 依赖（如果有的话）
# RUN pip install --no-cache-dir -r requirements.txt

# 可选：验证模型是否正确加载
RUN python3 -c "from ultralytics import YOLO; model = YOLO('/opt/nuclio/afs/boxstamp/nuclio/best.pt'); print('Model loaded successfully!')"