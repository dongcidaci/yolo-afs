## 依赖

- 基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- 使用自建数据集（非公开）

# yolo-afs

基于 YOLO 实现目标检测


# 构建镜像
docker -H tcp://172.16.22.25:2375 build -f Dockerfile -t cvat.pth.afs.boxstamp:latest .