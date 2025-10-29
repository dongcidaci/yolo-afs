import os


def rename_files_remove_prefix(directory_path):
    """
    读取目录下所有文件，将文件名中"-"及之前的文字去掉并重命名文件

    Args:
        directory_path (str): 目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录 {directory_path} 不存在")

    # 获取目录下的所有文件
    files = os.listdir(directory_path)
    renamed_count = 0

    for filename in files:
        # 检查文件名是否包含"-"
        if "-" in filename:
            # 获取"-"之后的部分作为新文件名
            new_filename = filename.split("-", 1)[1]

            # 构建完整的旧文件路径和新文件路径
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)

            # 检查新文件名是否已存在
            if os.path.exists(new_filepath):
                print(f"警告: 文件 {new_filename} 已存在，跳过重命名 {filename}")
                continue

            # 重命名文件
            try:
                os.rename(old_filepath, new_filepath)
                print(f"已重命名: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"重命名 {filename} 失败: {e}")

    print(f"重命名完成，共处理了 {renamed_count} 个文件")


# 使用示例
if __name__ == "__main__":
    # 替换为你的目录路径
    directory = "datasets/prepare/labels"
    rename_files_remove_prefix(directory)
