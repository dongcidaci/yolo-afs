import pandas as pd
import requests
import os
from urllib.parse import urlparse


def download_images_from_excel(
    excel_file_path, download_folder="downloads", max_size_mb=2
):
    """
    从Excel文件的第一列读取图片链接并下载图片，可限制图片大小

    Args:
        excel_file_path (str): Excel文件路径
        download_folder (str): 图片保存文件夹，默认为'downloads'
        max_size_mb (int): 最大图片大小限制(MB)，默认为2MB

    Returns:
        dict: 下载结果统计信息
    """
    # 创建下载目录
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # 读取Excel文件
    df = pd.read_excel(excel_file_path)

    # 获取第一列数据
    image_names = df.iloc[:, 0].dropna().tolist()  # 第一列，去除空值

    # 统计信息
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_urls = []

    # 下载图片
    for i, image_name in enumerate(image_names):
        url = "http://oss.fyxsyz.cn/" + image_name
        try:
            # 先发送HEAD请求检查文件大小
            head_response = requests.head(url, timeout=10)

            # 获取文件大小
            content_length = head_response.headers.get("content-length")

            if content_length:
                file_size_mb = int(content_length) / (1024 * 1024)
                if file_size_mb > max_size_mb:
                    print(
                        f"跳过大文件: {url} (大小: {file_size_mb:.2f}MB > {max_size_mb}MB)"
                    )
                    skipped_count += 1
                    continue

            # 发送HTTP请求下载图片
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # 检查请求是否成功

            # 如果HEAD请求没有获取到大小，这里再检查一次
            if not content_length:
                content_length = len(response.content)
                file_size_mb = content_length / (1024 * 1024)
                if file_size_mb > max_size_mb:
                    print(
                        f"跳过大文件: {url} (大小: {file_size_mb:.2f}MB > {max_size_mb}MB)"
                    )
                    skipped_count += 1
                    continue

            # 解析URL获取文件名
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)

            # 如果没有文件名，则使用索引作为文件名
            if not filename or "." not in filename:
                filename = f"image_{i+1}.jpg"

            # 完整保存路径
            file_path = os.path.join(download_folder, filename)

            # 保存图片
            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"成功下载: {url} -> {file_path} (大小: {file_size_mb:.2f}MB)")
            success_count += 1

        except Exception as e:
            print(f"下载失败: {url}, 错误: {str(e)}")
            failed_count += 1
            failed_urls.append(url)

    # 返回统计结果
    return {
        "total": len(image_names),
        "success": success_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "failed_urls": failed_urls,
    }


# 使用示例
if __name__ == "__main__":
    # 调用函数下载图片，限制大小为2MB
    result = download_images_from_excel(
        "/Users/jiangping/Downloads/签收图片.xls", "downloads", max_size_mb=2
    )

    # 打印统计结果
    print(f"\n下载完成:")
    print(f"总计: {result['total']} 张图片")
    print(f"成功: {result['success']} 张")
    print(f"失败: {result['failed']} 张")
    print(f"跳过: {result['skipped']} 张")
