import os


def read_images_and_generate_urls(directory_path, url_prefix, output_file="output.txt"):
    """
    读取指定目录下的图片文件，拼接URL前缀，并将结果输出到文件

    Args:
        directory_path (str): 图片目录路径
        url_prefix (str): URL前缀
        output_file (str): 输出文件名，默认为'output.txt'
    """
    # 支持的图片格式扩展名
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    # 获取目录下的所有文件
    image_urls = []

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录 {directory_path} 不存在")

    for filename in os.listdir(directory_path):
        # 获取文件扩展名并转为小写
        _, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            # 拼接完整的URL
            full_url = f"{url_prefix.rstrip('/')}/{filename}"
            image_urls.append(full_url)

    # 将URL写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        for url in image_urls:
            f.write(url + "\n")

    print(f"已成功处理 {len(image_urls)} 个图片文件，结果已保存到 {output_file}")


# 使用示例
if __name__ == "__main__":
    # 示例用法
    directory = "/Users/jiangping/Downloads/艾力斯审核图片/训练数据集/本地训练集"  # 替换为你的图片目录路径
    prefix = "http://oss.fyxsyz.cn/"
    read_images_and_generate_urls(directory, prefix)
